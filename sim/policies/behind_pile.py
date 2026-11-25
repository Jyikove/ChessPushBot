from baseline import RoutePolicy


class Policy(RoutePolicy):
    def __init__(self, setup):
        super().__init__(setup, behind=True)
