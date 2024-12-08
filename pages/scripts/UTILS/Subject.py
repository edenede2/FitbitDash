class Subject:
    __init__(self, name, project, date, runned, name_format, number, devices, path):
        self.name = name
        self.project = project
        self.date = date
        self.runned = runned
        self.name_format = name_format
        self.number = number
        self.devices = devices
        self.path = path
        
    def __str__(self):
        return f"Subject: {self.name}, Project: {self.project}, Date: {self.date}, Runned: {self.runned}, Name Format: {self.name_format}, Number: {self.number}, Devices: {self.devices}"
    
    def subject_to_json(self):
        return {
            "name": self.name,
            "project": self.project,
            "date": self.date,
            "runned": self.runned,
            "name_format": self.name_format,
            "number": self.number,
            "devices": self.devices,
            "path": self.path
        }
    
    def subject_update_json(self, data):
        self.name = data["name"]
        self.project = data["project"]
        self.date = data["date"]
        self.runned = data["runned"]
        self.name_format = data["name_format"]
        self.number = data["number"]
        self.devices = data["devices"]
        
    def subject_add_device(self, device):
        if device not in self.devices:
            self.devices.append(device)
            return True
        return False

        
    
        
    