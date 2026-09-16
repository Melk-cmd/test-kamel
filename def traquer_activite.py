def traquer_activite():
    historique_pas = []
    
    # --- PART 1: INPUT ---
    while True:
        pas = input("How many steps did you take? (or type 'fin' to stop): ")
        if pas == "fin":
            break  # Exit the while loop
            
        try:
            nombre_de_pas = int(pas)
        except ValueError:
            print("Invalid input error")
        else:
            historique_pas.append(nombre_de_pas)
            print(nombre_de_pas, "steps added to the history")
            break
            
    # --- PART 2: SUMMARY ---
    print("\n--- Summary of your walks ---")
    for releve in historique_pas:
        if releve >= 20000:
            print("Excellent:", releve, "steps. The goal is smashed!")
        elif releve >= 10000:
            print("Good:", releve, "steps. That's a good base.")
        else:
            print("Low:", releve, "steps. You'll have to catch up tomorrow.")

<<<<<<< HEAD
# Start the program
traquer_activite()
=======
# On lance le programme
traquer_activite()
##HELLO
>>>>>>> ac14e55 (Add hello comment)
