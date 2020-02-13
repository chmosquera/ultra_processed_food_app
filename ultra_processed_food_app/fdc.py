import requests
POST = requests.post
GET = requests.get
from os import environ
from sys import stderr

API_KEY_ENV_VAR = 'USDA_API_KEY'

SEARCH_ENDPOINT = 'https://api.nal.usda.gov/fdc/v1/search'
DETAIL_ENDPOINT = 'https://api.nal.usda.gov/fdc/v1/{fid}?api_key={apiKey}'

class SearchPageState:

    def __init__(self, responseData: dict):
        self.response = responseData
        self.currentPage = responseData['currentPage']
        self.totalPages = responseData['totalPages']
        self.totalFoods = responseData['totalHits']
        self.currentFoods = len(responseData['foods'])

    def update(self, responseData: dict):
        self.response = responseData
        self.currentPage = responseData['currentPage']
        self.currentFoods += len(responseData['foods'])

class FdcClient:

    def __init__(self, apiKey = None):
        if(apiKey == None):
            try:
                apiKey = environ[API_KEY_ENV_VAR]
            except KeyError as e:
                print(f"Could not retreive API key from environment variable '{API_KEY_ENV_VAR}'.", file = stderr)
                print("Please set this environment variable, or pass the API key to the constructor", file = stderr)
                raise KeyError("USDA API key not set")
        self.apiKey = apiKey
        self.params = {'api_key': apiKey}
        self.headers = {'Content-type':'application/json'}

    def search(self, generalSearchInput, pageLimit = None, productLimit = None, **kwargs):
        searchTerms = {'generalSearchInput': generalSearchInput}
        for key, arg in kwargs:
            searchTerms[key] = arg

        response = POST(SEARCH_ENDPOINT, headers=self.headers, params=self.params, json=searchTerms)
        response.raise_for_status()
        data = response.json()
        state = SearchPageState(data)
        if(pageLimit != None):
            state.totalPages = pageLimit
        if(productLimit != None):
            state.totalFoods = productLimit
        
        if('pageNumber' in searchTerms):
            for food in data['foods']:
                yield food
        else:
            searchTerms['pageNumber'] = 1

            foodsCovered = 0
            while(state.currentPage <= state.totalPages and foodsCovered <= state.totalFoods):
                for food in data['foods']:
                    yield food
                foodsCovered += len(data['foods'])
                
                searchTerms['pageNumber'] += 1
                response = POST(SEARCH_ENDPOINT, headers=self.headers, params=self.params, json=searchTerms)
                response.raise_for_status()
                data = response.json()
                state.update(data)

    def getFoodDetails(self, id):
        uri = DETAIL_ENDPOINT.format(fid=id, apiKey=self.apiKey)
        response = GET(uri)
        response.raise_for_status()
        return(GET(uri).json())
