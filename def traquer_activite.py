def traquer_activite():
    historique_pas = []
    
    # --- PARTIE 1 : LA SAISIE ---
    while True:
        pas = input("Combien de pas as-tu fait ? (ou tape 'fin' pour arrêter) : ")
        if pas == "fin":
            break # On sort de la boucle while
            
        try:
            nombre_de_pas = int(pas)
        except ValueError:
            print("Erreur de saisie invalide")
        else:
            historique_pas.append(nombre_de_pas)
            print(nombre_de_pas, "pas ajoutés à l'historique")
            break
            
    # --- PARTIE 2 : LE BILAN ---
    # La boucle for est maintenant reculée d'un cran vers la gauche
    print("\n--- Bilan de tes marches ---")
    for releve in historique_pas:
        # On utilise bien 'releve' dans les affichages
        if releve >= 20000:
            print("Excellent :", releve, "pas. L'objectif est pulvérisé !")
        elif releve >= 10000:
            print("Correct :", releve, "pas. C'est une bonne base.")
        else:
            print("Faible :", releve, "pas. Il faudra se rattraper demain.")

# On lance le programme
traquer_activite()