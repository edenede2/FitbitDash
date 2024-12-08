class Project:
    __init__(self, name, subjects, devices, students, path):
        self.name = name
        self.subjects = subjects
        self.devices = devices
        self.students = students
        self.path = path
        
    def __str__(self):
        return f"Project: {self.name}, Subjects: {self.subjects}, Devices: {self.devices}, Students: {self.students}"
    
    def project_to_json(self):
        return {
            "name": self.name,
            "subjects": self.subjects,
            "devices": self.devices,
            "students": self.students
        }
        
    def project_update_json(self, data):
        self.name = data["name"]
        self.subjects = data["subjects"]
        self.devices = data["devices"]
        self.students = data["students"]
        
    def project_add_subject(self, subject):
        if subject not in self.subjects:  
            self.subjects.append(subject)
            return True
        return False