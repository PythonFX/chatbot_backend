import httpx

from .llm_factory import get_profile


def get_azure_ad_token(profile_name: str = "azure") -> str:
    p = get_profile(profile_name)

    proxy_url = p.get("proxy_url")
    proxy_user = p.get("proxy_user")
    proxy_password = p.get("proxy_password")
    proxy = f"http://{proxy_user}:{proxy_password}@{proxy_url}" if proxy_url else None

    token_url = p.get("token_url")
    client_id = p.get("client_id")
    client_secret = p.get("client_secret")
    scope = "https://cognitiveservices.azure.com/.default"

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }

    try:
        with httpx.Client(proxy=proxy, verify=False) as client:
            response = client.post(token_url, data=data)

        if response.status_code == 200:
            token_info = response.json()
            return token_info.get("access_token")
        else:
            print("Failed to get token:", response.status_code)
            print("Response:", response.text)
            return ""
    except Exception as e:
        print(str(e))
        return ""
