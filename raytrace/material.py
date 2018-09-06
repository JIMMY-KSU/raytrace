class BasicMaterial:
    
    def __init__(self, color):
        self.color = color
        
    def getApparentColor(self, lighting):
        return self.color * lighting