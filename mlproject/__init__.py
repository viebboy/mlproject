try:
    import os
    from .constants import LOG_LEVEL
    if LOG_LEVEL == 'INFO':
        os.environ['LOGURU_LEVEL'] = 'INFO'
    elif LOG_LEVEL == 'DEBUG':
        os.environ['LOGURU_LEVEL'] = 'DEBUG'
except Exception:
    pass
