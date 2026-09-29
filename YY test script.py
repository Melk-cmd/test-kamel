class CompteBancaire:
    def __init__(self, titulaire, solde=0):
        self.titulaire = titulaire
        self.solde = solde

    def deposer(self, montant_deposer):
        self.solde += montant_deposer
        

    def retirer(self, montant_retirer):
        if self.solde < montant_retirer:
            print("erreur solde insuffisant")

        else:
            self.solde -= montant_retirer
            

    def afficher(self):
        print(f"{self.titulaire} avez {self.solde} euros sur votre compte bancaire")

compt1 = CompteBancaire("paul", 100)

compt1.deposer(100)
compt1.afficher()
compt1.retirer(250)
compt1.afficher()