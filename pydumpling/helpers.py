from traceback import print_tb, print_exception


def print_traceback_and_except(dumpling_result):
    exc_tb = dumpling_result["traceback"]
    except_extra = dumpling_result.get("exc_extra")
    exc_type = except_extra["exc_type"] if except_extra else None
    exc_value = except_extra["exc_value"] if except_extra else None
    if exc_type and exc_value:
        print_exception(exc_type, exc_value, exc_tb)
    else:
        print_tb(exc_tb)
