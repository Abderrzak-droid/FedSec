import matplotlib.pyplot as plt

# -------------------------------
# Premier graphique : Variation des méthodes Fed
# -------------------------------
# Données pour les 2 rounds, pour chaque méthode
# Données mises à jour pour le premier graphique
data_methods = {
    "FedAvg": {
        "2 rounds": {1: 0.8786620497703552, 2: 0.8955593943595886},
        "3 rounds": {1: 0.8892733573913574, 2: 0.905594003200531, 3: 0.900173008441925},
        "4 rounds": {1: 0.8909457921981812, 2: 0.8979815483093262, 3: 0.9126874327659606, 4: 0.9070357561111451}
    },
    "FedAdagrad": {
        "2 rounds": {1: 0.6767589211463928, 2: 0.5890426754951477},
        "3 rounds": {1: 0.7710495829582215, 2: 0.26833909740671513, 3: 0.6352364480495453},
        "4 rounds": {1: 0.7378892779350281, 2: 0.5011533975601197, 3: 0.7721453428268432, 4: 0.5754325330257416}
    },
    "FedMedian": {
        "2 rounds": {1: 0.916608989238739, 2: 0.9116493701934815},
        "3 rounds": {1: 0.8903114318847656, 2: 0.9102076053619385, 3: 0.905593991279602},
        "4 rounds": {1: 0.9232987284660339, 2: 0.9190888166427612, 3: 0.9134948134422303, 4: 0.9093425631523132}
    }
}

# Création d'une figure avec trois sous-graphiques (un par algorithme)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (method, runs) in zip(axes, data_methods.items()):
    for run_label, rounds_dict in runs.items():
        rounds = list(rounds_dict.keys())
        accuracies = list(rounds_dict.values())
        ax.plot(rounds, accuracies, marker='o', linestyle='-', label=run_label)
    ax.set_title(method)
    ax.set_xlabel("Round")
    ax.grid(True)
    ax.legend()
    
axes[0].set_ylabel("Accuracy")
plt.suptitle("Comparaison des algorithmes Fed selon le nombre de rounds", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# -------------------------------
# Deuxième graphique : Variation du nombre de rounds (FedMedian)
# -------------------------------
# Données pour FedMedian avec 3, 4 et 5 rounds
rounds_data = {
    "3 Rounds": {1: 0.9223183393478394, 2: 0.9136101484298706, 3: 0.9118223786354065},
    "4 Rounds": {1: 0.9083621621131897, 2: 0.9057093501091004, 3: 0.909400224685669, 4: 0.9069780826568603},
    "5 Rounds": {1: 0.9111303329467774, 2: 0.9149942398071289, 3: 0.9132064700126648, 4: 0.9129757761955262, 5: 0.9134371399879455}
}

plt.figure(figsize=(8, 6))
for label, rounds_dict in rounds_data.items():
    rounds = list(rounds_dict.keys())
    accuracies = list(rounds_dict.values())
    plt.plot(rounds, accuracies, marker='o', label=label)

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy en fonction du nombre de rounds (FedMedian)")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------------
# Troisième graphique : Comparaison entre différents clients (FedMedian)
# -------------------------------
# Données pour différents clients sur 3 rounds
clients_data = {
    "Client6": {1: 0.8999423185984293, 2: 0.9152249197165171, 3: 0.9144175251324972},
    "Client7": {1: 0.8687929000173297, 2: 0.919603203024183, 3: 0.90114768913814},
    "Client8": {1: 0.7731203883886337, 2: 0.9047509282827377, 3: 0.908440962433815},
    "Client9": {1: 0.7309000723891788, 2: 0.8949432041909959, 3: 0.8943666021029154},
    "Client10": {1: 0.7386966586112976, 2: 0.9043829262256622, 3: 0.8952710628509521}
}

plt.figure(figsize=(8, 6))
for client, rounds_dict in clients_data.items():
    rounds = list(rounds_dict.keys())
    accuracies = list(rounds_dict.values())
    plt.plot(rounds, accuracies, marker='o', label=client)

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy par round pour différents clients (FedMedian)")
plt.legend()
plt.grid(True)
plt.show()
