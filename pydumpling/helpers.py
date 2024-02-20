import linecache


def print_traceback_and_except(dumpling_result):
    traceback_obj = dumpling_result["traceback"]
    except_extra = dumpling_result["exc_extra"]
    except_type = except_extra["exc_type"]
    except_value = except_extra["exc_value"]

    print("Traceback (most recent call last):")
    while traceback_obj:
        frame = traceback_obj.tb_frame
        lineno = traceback_obj.tb_lineno
        filename = frame.f_code.co_filename
        function_name = frame.f_code.co_name
        line = linecache.getline(filename, lineno).strip()
        mid_index = len(line) // 2

        print(f"File \"{filename}\", line {lineno}, in {function_name}")
        print(f"  {line}")
        print(" " * 4 + "~" * mid_index + "^" + "~" * mid_index)
        traceback_obj = traceback_obj.tb_next

        if traceback_obj is None:
            break
    print(f"{except_type.__name__}: {except_value}")
