import service.twitter_service as twitter_service

class ApiPool:
    def __init__(self, api_credentials):
        # TODO implement command pattern to avoid Twitter API's max retries limit exceeded'
        self.api_credentials = api_credentials
        self.pool = []

    def create(self):
        return twitter_service.create_api(self.api_credentials.pop()) if len(self.api_credentials) else None

    def get(self):
        # get first free API instance or create new one
        return self.pool.pop() if len(self.pool) else self.create()

    def release(self, api):
        self.pool.append(api)