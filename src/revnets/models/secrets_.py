from dataclasses import dataclass


@dataclass
class ApiSecrets:
    token: str = "token"


@dataclass
class Secrets:
    api: ApiSecrets
