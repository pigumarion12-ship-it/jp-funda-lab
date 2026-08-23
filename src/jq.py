"""J-Quants API V2 client wrapper (公式 jquants-api-client の ClientV2 を使用)."""
import os
import jquantsapi


def get_client() -> "jquantsapi.ClientV2":
    key = os.environ.get("JQUANTS_API_KEY")
    if not key:
        raise RuntimeError("JQUANTS_API_KEY が設定されていません")
    return jquantsapi.ClientV2(api_key=key)
