import random
import numpy as np
from scipy.spatial import cKDTree as KDTree
from copy import deepcopy
from numbers import Number


class NoveltyArchive:

    def __init__(self, genomes: list[list[Number]], k: int, max_archive_dim: int = -1,
                 remove_less_innovative: bool = False,
                 innovative_threshold: float = 0.20):
        '''
        Classe che permette calcolare la novelty di ogni genoma.  Un genoma all'interno dell'archivio è singolo
        per cui eventuali duplicati sono rimossi.
        :param genomes: Lista di genomi, dove ogni genoma è una lista di valori numerici. Ogni lista deve avere la stessa lunghezza. ES: genomes = [[3,2,1], [1,2,3]]
        :param k: Parametro che indica il numero di vicini per il quale calcolare la distanza. ES: k=15
        :param max_archive_dim: Dimensioni massime dell'archivio, superate le quali si rimuovono gli elementi meno nuovi. Se questo valore è <=1 l'archivio cresce senza vincoli
        :param remove_less_innovative: Se vero, i genomi che non introducono almeno una percentuale di threshold di innovazione, vengono rimossi
        :param innovative_threshold: Valore numerico in [0, 1] che specifica in che percentuale (rispetto il più innovativo) deve essere differente.
        '''
        # Numero di genomi per i quali eseguire la comparazione
        self.__max_archive_dim = max_archive_dim
        self.__k = k
        self.__genomes = []
        self.__kd_tree = None
        self.__is_new_archive = True
        self.__innovative_threshold = innovative_threshold
        self.__remove_less_innovative = remove_less_innovative
        if len(genomes) > 0:
            if self.__check_genomes(genomes=genomes):
                self.add_genomes_to_archive(new_genomes=genomes)
            else:
                raise Exception("Lista di genomi non valida")

    ### Private Methods ###
    def __check_genomes(self, genomes: list[list[Number]]) -> bool:
        '''
        Controlla la lista di genomi. Devono avere stessa lunghezza diversa da 0 e devono superare il
        controllo del singolo genoma
        :param genomes: Lista di genomi
        :return: True se il check è passato False altrimenti
        '''
        try:
            check_len = len(genomes) > 0
            check_gene_nb = len(set([len(genome) for genome in genomes])) == 1
            check_genome = False not in [self.__check_genome(genome=genome) for genome in genomes]
            return check_len and check_genome and check_gene_nb
        except Exception as e:
            print(e)
            return False

    def __check_new_genomes(self, new_genomes: list[list[Number]]) -> bool:
        '''
        Controlla la nuova lista di genomi che si vuole aggiungere. Deve essere consistente rispetto
        quelli già aggiunti.
        :param new_genomes: Lista di nuovi genomi.
        :return: True se il check è passato False altrimenti
        '''
        # Nel caso in cui ho permesso una inizializzazione vuota non faccio il check coi vecchi genomi
        if self.__is_new_archive:
            return True
        else:
            try:
                genome_len = len(self.__genomes[0])
                check_new_genomes = self.__check_genomes(genomes=new_genomes)
                check_new_genomes_len = False not in [len(new_genome) == genome_len for new_genome in new_genomes]
                return check_new_genomes_len and check_new_genomes
            except Exception as e:
                print(e)
                return False

    def __check_genome(self, genome: list[Number]) -> bool:
        '''
        Controlla il singolo genoma. Deve essere una lista di valori numerici di lunghezza maggiore di zero
        :param genome: Genoma da controllare
        :return: True se il controllo è superato, False altrimenti
        '''
        try:
            check_type = type(genome) is list
            check_len = len(genome) > 0
            check_val = False not in [True if type(gene) in [int, float] or np.char.isnumeric(str(gene)) else False
                                      for gene in genome]
            return check_val and check_type and check_len
        except Exception as e:
            print(e)
            return False

    def __reset_archive(self) -> None:
        self.__genomes = []
        self.__kd_tree = None
        self.__is_new_archive = True

    def __get_less_novel_genomes(self, novel_check_type: str) -> list[list[Number]]:
        '''
        Restituisce una lista di genomi che possono essere rimossi
        :param novel_check_type: Valore che indica il tipo di studio. Se novel_check_type=dimension rimuove quelli meno innovativi che eccedono le dimensioni massime dell'archivio, se novel_check_type=threshold rimuove tutti i genomi che differiscono di un valore minore del threshold dato indipendentemente dalle dimensioni
        :return:
        '''
        less_novels = []
        novel_check_type = novel_check_type.lower()
        if novel_check_type == "dimension":
            nb_genomes_to_remove = self.get_size() - self.__max_archive_dim
            if nb_genomes_to_remove > 0:
                genomes, novelties = self.get_novelty_value(genomes=self.get_genomes())
                less_novels = list(zip(genomes, novelties))
                less_novels = sorted(less_novels, key=lambda x: x[1])
                less_novels = less_novels[0:nb_genomes_to_remove]
                less_novels = [genome[0] for genome in less_novels]
        elif novel_check_type == "threshold":
            genomes, novelties = self.get_novelty_value(genomes=self.get_genomes())
            compared = []
            novels = list(zip(genomes, novelties))
            novels = sorted(novels, key=lambda x: x[1], reverse=True)
            for novel_1 in novels:
                for novel_2 in novels:
                    genome_1 = novel_1[0]
                    genome_2 = novel_2[0]
                    comparing_1 = (tuple(genome_1), tuple(genome_2))
                    comparing_2 = (tuple(genome_2), tuple(genome_1))
                    if genome_1 != genome_2 and comparing_1 not in compared and comparing_2 not in compared:
                        novel_val_1 = novel_1[1]
                        novel_val_2 = novel_2[1]
                        variation = abs(novel_val_1 - novel_val_2) / novel_val_1
                        if variation < self.__innovative_threshold:
                            if random.uniform(0, 1) < 0.5:
                                less_novels.append(genome_1)
                            else:
                                less_novels.append(genome_2)
                            compared.append(comparing_1)
                            compared.append(comparing_2)
        else:
            print(f"Tipo di rimozione {novel_check_type} non riconosciuta")
        return less_novels

    def __remove_duplicates_genome(self, genomes: list[list[Number]]) -> list[list[Number]]:
        '''
        Funzione che rimuove i duplicati da un dato insieme di genomi, e quelli già presenti nell'archivio
        :param genomes: Insieme di nuovi genomi
        :return: Lista di genomi senza duplicati
        '''
        cleaned_genomes = []
        # Removing if in archive
        for genome in genomes:
            if self.check_if_genome_is_present(genome=genome) is False: cleaned_genomes.append(genome)
        # Removing if in itself
        to_delete = []
        for genome in cleaned_genomes:
            # Avoid duplicates
            if cleaned_genomes.count(genome) > 1 and genome not in to_delete: to_delete.append(genome)
        # Rimuovo tutte le occorrenze duplicate
        cleaned_genomes = [genome for genome in cleaned_genomes if genome not in to_delete]
        # Le reimmetto nell'archivio
        for deleted in to_delete:
            cleaned_genomes.append(deleted)
        return cleaned_genomes

    ### Public Methods ###
    def add_genomes_to_archive(self, new_genomes: list[list[Number]]) -> None:
        '''
        Aggiunge una lista di genomi all'archivio
        :param new_genomes: Lista di nuovi genomi
        :return: None
        '''
        new_genomes = deepcopy(new_genomes)
        if self.__check_new_genomes(new_genomes=new_genomes):
            new_genomes = self.__remove_duplicates_genome(genomes=new_genomes)
            self.__genomes += new_genomes
            # Struttura dati contenente i genomi nello spazio
            self.__kd_tree = KDTree(self.__genomes)
            if self.__max_archive_dim > 1:
                less_novel_genomes = self.__get_less_novel_genomes(novel_check_type="dimension")
                self.remove_genomes(genomes_to_remove=less_novel_genomes, refresh_model=True)
            if self.__remove_less_innovative:
                less_novel_genomes = self.__get_less_novel_genomes(novel_check_type="threshold")
                self.remove_genomes(genomes_to_remove=less_novel_genomes, refresh_model=True)
            self.__is_new_archive = False
        else:
            if self.__is_new_archive:
                raise Exception("Lista di genomi non valida")
            else:
                raise Exception("Lista di nuovi genomi non valida")

    def get_size(self) -> int:
        '''
        Funzione che calcola la lunghezza dell'archivio
        :return: lunghezza dell'archivio
        '''
        return len(self.__genomes)

    def get_novelty_value(self, genomes: list[list[Number]]) -> (list[list[Number]], list[float]):
        '''
        Funzione che calcola la distanza media con i primi k genomi ordinati in ordine crescente
        :param genomes: Lista di genomi
        :return: Distanza media del genoma
        '''
        genomes_res = []
        values = []
        if self.__kd_tree is not None:
            for genome in genomes:
                if self.__check_genome(genome=genome):
                    current_size = self.get_size()
                    if self.__k < current_size:
                        nb_neighbours = self.__k
                    else:
                        print(f"Dimensioni correnti dell'archivio: {current_size}")
                        print(f"Valore di k selezionato: {self.__k}")
                        print(f"Sarà utilizzato k={current_size}")
                        nb_neighbours = current_size
                    if nb_neighbours > 1:
                        # p=2 è distanza euclidea
                        distances, _ = self.__kd_tree.query(np.array(genome), nb_neighbours, p=2)
                        distances = list(distances)
                        # Calcolo la distanza media tra il genoma con i k genomi piu vicini (avendo utilizzato sort)
                        value = sum(distances[:nb_neighbours + 1]) / nb_neighbours
                        value = round(value, 10)
                    else:
                        value = 0.0
                    genomes_res.append(genome)
                    values.append(value)
                else:
                    raise Exception("Genoma non valido")
        else:
            raise Exception("Can't compute novelty value. Are genomes populated?")
        return genomes_res, values

    def clean_archive(self):
        self.__reset_archive()

    def check_if_genome_is_present(self, genome: list[Number]) -> bool:
        '''
        Questo metodo controlla se un genoma è già presente nell'archivio
        :param genome: Genoma da controllare
        :return: True se presente nell'archivio, False altrimenti
        '''
        return genome in self.__genomes

    def remove_genomes(self, genomes_to_remove: list[list[Number]], refresh_model: bool = True) -> None:
        '''
        Quest metodo rimuove i genomi dall'archivio
        :param genomes_to_remove: Lista di genomi da rimuovere
        :param refresh_model: Se True, viene ricostruito il modello scipy per il calcolo della distanza. Se False non viene ricostruito e pur non avendo i genomi nella lista, continueranno ad essere presenti nel modello.
        :return: None
        '''
        removed = False
        for genome in genomes_to_remove:
            if self.check_if_genome_is_present(genome):
                self.__genomes.remove(genome)
                removed = True
        if self.get_size() > 0:
            if refresh_model and removed:
                self.__kd_tree = KDTree(self.__genomes)
        else:
            self.__reset_archive()
        return None

    def get_genomes(self, convert_to_numpy: bool = False) -> list[list[Number]]:
        if not convert_to_numpy:
            genomes = deepcopy(self.__genomes)
        else:
            genomes = deepcopy(self.__genomes)
            genomes = np.array([np.array(genome) for genome in genomes])
        return genomes

    def get_most_novel_genome(self, number: int = 1) -> (list[list[Number]], float):
        best_genomes = []
        best_novelties = []
        genomes, novelties = self.get_novelty_value(genomes=self.get_genomes())
        novels = list(zip(genomes, novelties))
        novels = sorted(novels, key=lambda x: x[1], reverse=True)
        for i in range(0, number):
            elem = novels[i]
            best_genomes.append(elem[0])
            best_novelties.append(elem[1])
        return best_genomes, novelties

    def __str__(self) -> str:
        if self.get_size() > 0 and self.__kd_tree is not None:
            res = ""
            genomes, novelties = self.get_novelty_value(genomes=self.get_genomes())
            archive_str_rep = list(zip(genomes, novelties))
            archive_str_rep = sorted(archive_str_rep, key=lambda x: x[1], reverse=True)
            for elem in archive_str_rep:
                data = f"Genome={elem[0]} --> Novelty={elem[1]}\n"
                res += data
        else:
            res = "Empty archive"
        return res

# example = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 7, 1], [6, 1, 2, 3], [1, 5, 3, 1], [1, 5, 3, 7], [12, 52, 31, 7]]
# na = NoveltyArchive(genomes=[], k=2, max_archive_dim=5)
# na.add_genomes_to_archive(new_genomes=example)
# print(na.get_size())
# print(na.get_genomes())
# print(str(na))
# genome, novel = na.get_most_novel_genome()
# print(genome, novel)
