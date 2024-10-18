import time 

class Logger():
    filepath = None
    print_to_console = True
    line_num = 0
    #initalize the logger
    def __init__(self, print_to_console: bool = True):
        #create a log file named Log_<timestamp>.txt
        t = ""
        year = time.localtime().tm_year
        month = time.localtime().tm_mon
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month = months[month-1]
        day = time.localtime().tm_mday
        hour = time.localtime().tm_hour
        minute = time.localtime().tm_min
        second = time.localtime().tm_sec
        t = f"{year}_{month}{day}_{hour}:{minute}:{second}"
        self.filepath = f"logs/Log_{t}.txt"
        self.print_to_console = print_to_console
        self.create_log_file()


    def create_log_file(self):
        #create a log file
        with open(self.filepath, "w") as file:
            print("Log file created, print to console: " + str(self.print_to_console))
            file.write("[" + str(self.line_num) + "] Log file '"+self.filepath+"' created, print to console: " + str(self.print_to_console) + "\n")
            self.line_num += 1

    def log(self, message):
        #log the message to the log file
        with open(self.filepath, "a") as file:
            file.write("[" + str(self.line_num) + "] " + message + "\n")
            self.line_num += 1
            if self.print_to_console:
                print(message)


