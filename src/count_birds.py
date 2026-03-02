import json
import glob

def main():

    class_files = glob.glob("data/models/*task-species*_classes.json")

    if not class_files:
        print("Konnte keine classes.json finden. Hast du Modelle trainiert?")
        return

    file_to_read = class_files[0]
    print(f"Lese Klassen aus: {file_to_read}\n")

    with open(file_to_read, 'r') as f:
        classes = json.load(f)['classes']

    non_bird_keywords =[
        'cricket', 'frog', 'toad', 'cicada', 'bat', 'grasshopper',
        'katydid', 'roesel', 'mouse', 'mole', 'shrew', 'squirrel',
        'deer', 'fox', 'wolf', 'marmot', 'locust', 'insect', 'bush-cricket'
    ]

    birds = []
    distractors =[]

    for species in classes:

        if any(keyword in species.lower() for keyword in non_bird_keywords):
            distractors.append(species)
        else:
            birds.append(species)

    print("="*50)
    print("  PIEPMATZ POLIZEI - DATENSATZ ANALYSE")
    print("="*50)
    print(f"Gesamte Klassen im Modell:      {len(classes)}")
    print(f"➔ Echte Vogelarten:             {len(birds)}")
    print(f"➔ Insekten/Amphibien (Filter):  {len(distractors)}")
    print("="*50)

    print("\nBeispiele für herausgefilterte 'Distractor Classes':")
    for d in distractors[:15]:
        print(f" - {d}")

    print("\nDiese Störgeräusche wird die KI im Wald nun erfolgreich ignorieren! ")

if __name__ == "__main__":
    main()
