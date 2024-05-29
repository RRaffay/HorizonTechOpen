from sec_api import QueryApi, ExtractorApi
import pandas as pd
from django.conf import settings
import requests
import json
from .models import SECFiling
from user_service.models import Stock


import logging

logger = logging.getLogger(__name__)


def query_filings(tickers, form_type):
    api_key = settings.SEC_API_KEY
    queryApi = QueryApi(api_key=api_key)

    ticker_query = " OR ".join([f"ticker:{ticker}" for ticker in tickers])

    # Change the formType query to use the passed form_type
    # Currently getting the latest 10 filings for each ticker may need to change according to use case
    query = {
        "query": {
            "query_string": {"query": f'({ticker_query}) AND formType:"{form_type}"'}
        },
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    stock = Stock.objects.get(ticker=tickers[0])

    try:
        response = queryApi.get_filings(query)
        filings = response.get("filings", [])
        for filing in filings:
            # Check if filing already exists in database
            if not SECFiling.objects.filter(accessionNo=filing["accessionNo"]).exists():
                # Assuming all required fields are in the response, otherwise adjust accordingly
                SECFiling.objects.create(
                    accessionNo=filing["accessionNo"],
                    # This needs to be the same the stock ticker in the database for the foreign key to work
                    stock=stock,
                    cik=filing["cik"],
                    ticker=filing["ticker"],
                    companyName=filing["companyName"],
                    companyNameLong=filing["companyNameLong"],
                    formType=filing["formType"],
                    description=filing["description"],
                    linkToFilingDetails=filing["linkToFilingDetails"],
                    linkToTxt=filing["linkToTxt"],
                    linkToHtml=filing["linkToHtml"],
                    linkToXbrl=filing.get(
                        "linkToXbrl", None
                    ),  # assuming this might be optional
                    filedAt=pd.to_datetime(filing["filedAt"]),
                    periodOfReport=pd.to_datetime(filing["periodOfReport"])
                    if "periodOfReport" in filing
                    else None,
                    effectivenessDate=pd.to_datetime(filing["effectivenessDate"])
                    if "effectivenessDate" in filing
                    else None,
                    registrationForm=filing.get("registrationForm", None),
                    referenceAccessionNo=filing.get("referenceAccessionNo", None),
                )
                logger.info(
                    f"Saved {filing['formType']} for {filing['ticker']} to database"
                )

        logger.info(f"Saved {len(filings)} filings to database")

    except Exception as e:
        logger.exception(f"Error fetching or saving filings: {e}")
        print(f"Error fetching or saving filings: {e}")
        return pd.DataFrame()  # return an empty DataFrame in case of an error

    return pd.DataFrame.from_records(filings)


def extract_key_items(tickers, sections, form_type="10-K"):
    all_ticker_data = {}
    api_key = settings.SEC_API_KEY

    extractorApi = ExtractorApi(api_key=api_key)

    for ticker in tickers:
        # Fetch the latest filing from the local database
        ticker_filings = SECFiling.objects.filter(
            ticker=ticker, formType__in=["10-K", "10-Q", "8-K"]
        ).order_by("-filedAt")

        # If no filings exist for the ticker, query the SEC API and add the filing to the database
        if not ticker_filings.exists():
            logger.info(f"No filings found for {ticker}")
            query_filings(tickers, form_type)
            logger.info(f"Added filing for {ticker} to database")

        # Fetch the latest filing from the local database
        ticker_filings = SECFiling.objects.filter(
            ticker=ticker, formType__in=["10-K", "10-Q", "8-K"]
        ).order_by("-filedAt")

        filing = ticker_filings.first()
        linkToHtml = filing.linkToHtml

        filing_date = filing.filedAt
        filing_type = filing.formType
        filing_description = filing.description

        ticker_data = {}
        ticker_data["text"] = {}
        for section in sections:
            ticker_data["text"][section] = extractorApi.get_section(
                linkToHtml, section, "text"
            )

        all_ticker_data[ticker] = ticker_data
        all_ticker_data[ticker]["filing_date"] = filing_date
        all_ticker_data[ticker]["filing_type"] = filing_type
        all_ticker_data[ticker]["filing_description"] = filing_description

    return all_ticker_data, filing


def fetch_financial_statements(ticker, form_type="10-K", size="10"):
    sec_api_key = settings.SEC_API_KEY

    queryApi = QueryApi(api_key=sec_api_key)

    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND formType:"{form_type}"',
            }
        },
        "from": "0",
        "size": size,
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    response = queryApi.get_filings(query)
    metadata = pd.DataFrame.from_records(response["filings"])
    link_to_10Q = metadata["linkToHtml"][0]

    # XBRL-to-JSON converter API endpoint
    xbrl_converter_api_endpoint = "https://api.sec-api.io/xbrl-to-json"
    final_url = (
        xbrl_converter_api_endpoint
        + "?htm-url="
        + link_to_10Q
        + "&token="
        + sec_api_key
    )

    # make request to the API
    response = requests.get(final_url)
    xbrl_json = json.loads(response.text)

    return xbrl_json


def get_financial_statements(ticker):
    xbrl_json = fetch_financial_statements(ticker)

    income_statement = get_income_statement(xbrl_json)
    balance_sheet = get_balance_sheet(xbrl_json)
    cash_flow_statement = get_cash_flow_statement(xbrl_json)

    return income_statement, balance_sheet, cash_flow_statement


# Your existing get_income_statement, get_balance_sheet, and get_cash_flow_statement functions remain the same
def get_income_statement(xbrl_json):
    income_statement_store = {}

    # iterate over each US GAAP item in the income statement
    for usGaapItem in xbrl_json["StatementsOfIncome"]:
        values = []
        indicies = []

        for fact in xbrl_json["StatementsOfIncome"][usGaapItem]:
            # only consider items without segment. not required for our analysis.
            if "segment" not in fact:
                index = fact["period"]["startDate"] + "-" + fact["period"]["endDate"]
                # ensure no index duplicates are created
                if index not in indicies:
                    values.append(fact["value"])
                    indicies.append(index)

        income_statement_store[usGaapItem] = pd.Series(values, index=indicies)

    income_statement = pd.DataFrame(income_statement_store)
    # switch columns and rows so that US GAAP items are rows and each column header represents a date range
    return income_statement.T


def get_balance_sheet(xbrl_json):
    balance_sheet_store = {}

    for usGaapItem in xbrl_json["BalanceSheets"]:
        values = []
        indicies = []

        for fact in xbrl_json["BalanceSheets"][usGaapItem]:
            # only consider items without segment.
            if "segment" not in fact:
                index = fact["period"]["instant"]

                # avoid duplicate indicies with same values
                if index in indicies:
                    continue

                # add 0 if value is nil
                if "value" not in fact:
                    values.append(0)
                else:
                    values.append(fact["value"])

                indicies.append(index)

            balance_sheet_store[usGaapItem] = pd.Series(values, index=indicies)

    balance_sheet = pd.DataFrame(balance_sheet_store)
    # switch columns and rows so that US GAAP items are rows and each column header represents a date instant
    return balance_sheet.T


def get_cash_flow_statement(xbrl_json):
    cash_flows_store = {}

    for usGaapItem in xbrl_json["StatementsOfCashFlows"]:
        values = []
        indicies = []

        for fact in xbrl_json["StatementsOfCashFlows"][usGaapItem]:
            # only consider items without segment.
            if "segment" not in fact:
                # check if date instant or date range is present
                if "instant" in fact["period"]:
                    index = fact["period"]["instant"]
                else:
                    index = (
                        fact["period"]["startDate"] + "-" + fact["period"]["endDate"]
                    )

                # avoid duplicate indicies with same values
                if index in indicies:
                    continue

                if "value" not in fact:
                    values.append(0)
                else:
                    values.append(fact["value"])

                indicies.append(index)

        cash_flows_store[usGaapItem] = pd.Series(values, index=indicies)

    cash_flows = pd.DataFrame(cash_flows_store)
    return cash_flows.T
