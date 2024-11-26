class Kullanici():
    def __init__(self, isim, yas):
        self.isim = isim
        self.__yas = yas
    
    def yasi_goster(self):
        print(f"{self.isim} yaş notu: {self.__yas}")

kullanici = Kullanici("Mehmet", 65)
kullanici.__sinavNotu = 55

kullanici.yasi_goster()