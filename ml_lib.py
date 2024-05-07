import random

class stats:
    def __init__(self, data:list):
        self.data = data

    def z_score(self) -> list:

        tdata = list(zip(*self.data)) #transpose to deal with columns

        z_data = []

        for i in tdata[:-1]:
            u = sum(i)/len(i) #mean
            s = (sum([(x - u)**2 for x in i])/len(i))**(1/2) #std (sqrt of pop var)
            z_data += [[(x-u)/s for x in i]]  #apply z-score to each datapoint
        
        z_data += [tdata[-1]]
        z_data = list(map(list, list(zip(*z_data))))  #transpose and map to original 2d-matrix dimensions

        return z_data


class k_nearest_neighbors:
    def __init__(self, data:list, nn:int, fold:int):
        self.data = data
        self.nn = nn
        self.fold = fold


    def classifier(self, train_set:list, test_set:list) -> list|list:
        train_labels = [int(x[-1]) for x in train_set]
        test_labels = [int(x[-1]) for x in test_set]

        predicted_labels = []

        for _,j in enumerate(test_set):
            dist_list = []
            for _,b in enumerate(train_set):
                dist = []
                for k,_ in enumerate(test_set[0]):
                    dist += [(j[k] - b[k])**2] #quadratic diff
                dist_list += [(sum(dist))**(1/2)] #euclidean distance
            predicted_labels += [self.find_nn_class(self.nn, train_labels, dist_list)]

        return predicted_labels, test_labels
        

    def find_nn_class(self, nn:int, train_labels:list, dist_list:list) -> int: #arbitrary nn due to recursion
        nn_list = []
        original_dist_list = dist_list.copy()
        for _ in range(nn):
            nn_list += [train_labels[(dist_list.index(min(dist_list)))]]
            dist_list.pop(dist_list.index(min(dist_list)))

        classes = dict()
        for index in set(nn_list):
            counter = 0
            for c in nn_list:
                if index == c:
                    counter += 1
            classes.update({index: counter})

        if len(classes.values()) > 1 & len(set(list(classes.values()))) == 1:  #find draws
            return self.find_nn_class(nn-1, train_labels, original_dist_list)  #recursion until unanimity
        else: #no draw
            return (list(classes.keys())[list(classes.values()).index(max(list(classes.values())))])


    def cross_validation(self) -> list|list:
        shuffled_data = self.data.copy()
        random.shuffle(shuffled_data)
        test_size = int(len(shuffled_data)/self.fold)

        predicted_folds = []
        original_folds = []

        for i in range(self.fold):
            test_set = shuffled_data[i*test_size:(i+1)*test_size]
            train_set = [item for item in shuffled_data if item not in test_set]
            predicted, original = self.classifier(train_set, test_set)
            predicted_folds += [predicted]
            original_folds += [original]
        
        return predicted_folds, original_folds


    def run(self) -> float:
        return self.cross_validation()
    
class metrics:
    def __init__(self, predicted:list, original:list, fold:list):
        self.predicted = predicted
        self.original = original
        self.fold = fold

    def accuracy(self) -> float:
        if len(self.predicted) == len(self.original) == self.fold:
            mean_acc = []

            for i,_ in enumerate(self.predicted):
                mean_acc += [sum([1 if x == y else 0 for x,y in zip(self.predicted[i],self.original[i])])/len(self.predicted[i])]
            
            return round(sum(mean_acc)/len(mean_acc),3) #round up to 3 digits
        else:
            raise ValueError("Input wrongly shaped")