from sklearn import svm, tree, naive_bayes
import csv
import random
from statistics import mean


def gather_data():
    with open('Elements.csv') as csvfile:
        element_data = []
        element_data_csv = csv.reader(csvfile, delimiter=',')
        for row in element_data_csv:
            element_data.append(row)
        i = 0
        index = 0
        for row in element_data:
            i = 0
            for word in row:
                row[i] = word.strip(' ')
                if '(' in word:
                    word = word[:-3]
                    word = word.strip(' ')
                    row[i] = word
                if '[' in word:
                    letter_list = []
                    for letter in word:
                        letter_list.append(letter)
                    letter_list.pop(0)
                    i2 = 0
                    for letter in letter_list:
                        if letter == '[' or letter == ']':
                            letter_list.pop(i2)
                        i2 += 1
                    row[i] = ''.join(letter_list)
                i += 1
            index += 1
            i += 1
        return element_data


class Element:
    """Class that stores the Atomic Number, The proton to neutron ratio, the protons and the neutrons"""
    PtoN_Ratio = 0

    def __init__(self, p, n):
        self.p = p
        self.n = n
        Element.PtoN_Ratio = n / p


def assign_data():
    element_data = gather_data()
    element_list = []
    i = 0
    PtoNList = []
    for element in element_data:
        element_list.append(Element(float(element[0]), (float(element[2]) - float(element[0]))))
        PtoNList.append(element_list[i].PtoN_Ratio)
        i += 1
    stable_list = []
    alpha_list = []
    beta_list = []
    positron_list = []
    i2 = 0
    for element in element_list:
        if element.p > 82:
            alpha_list.append([element.p, PtoNList[i2]])
        if 1 <= PtoNList[i2] <= 1.5 and element.p <= 82:
            stable_list.append([element.p, PtoNList[i2]])
        if PtoNList[i2] > 1.5 and element.p <= 82:
            beta_list.append([element.p, PtoNList[i2]])
        if PtoNList[i2] < 1 and element.p <= 82:
            positron_list.append([element.p, PtoNList[i2]])
        i2 += 1
    positron_list.pop(0)
    stable_list_predict = random.sample(stable_list, 10)
    i = 0
    for element in stable_list:
        if element in stable_list_predict:
            stable_list.pop(i)
        i += 1

    alpha_list_predict = random.sample(alpha_list, 10)
    i = 0
    for element in alpha_list:
        if element in alpha_list_predict:
            alpha_list.pop(i)
        i += 1

    beta_list_predict = random.sample(beta_list, 10)
    i = 0
    for element in beta_list:
        if element in beta_list_predict:
            beta_list.pop(i)
        i += 1

    positron_list_predict = random.sample(positron_list, 10)
    i = 0
    for element in positron_list:
        if element in positron_list_predict:
            positron_list.pop(i)
        i += 1

    return (stable_list, stable_list_predict, alpha_list, alpha_list_predict, beta_list, beta_list_predict,
            positron_list, positron_list_predict)


def ml_test():
    stable_list = assign_data()[0]
    stable_list_predict = assign_data()[1]
    alpha_list = assign_data()[2]
    alpha_list_predict = assign_data()[3]
    beta_list = assign_data()[4]
    beta_list_predict = assign_data()[5]
    positron_list = assign_data()[6]
    positron_list_predict = assign_data()[7]

    X = stable_list + alpha_list + beta_list + positron_list
    y = []
    for element in stable_list:
        y.append('Stable')
    for element in alpha_list:
        y.append('Alpha')
    for element in beta_list:
        y.append('Beta')
    for element in positron_list:
        y.append('Positron')

    def svm_test():
        clf = svm.LinearSVC()
        clf.fit(X, y)
        stable_results = clf.predict(stable_list_predict)
        alpha_results = clf.predict(alpha_list_predict)
        beta_results = clf.predict(beta_list_predict)
        positron_results = clf.predict(positron_list_predict)
        Stable_score = 0
        for result in stable_results:
            if result == 'Stable':
                Stable_score += 1

        Stable_score_f = (Stable_score/len(stable_results))
        Stable_score_p = Stable_score_f * 100

        Alpha_score = 0
        for result in alpha_results:
            if result == 'Alpha':
                Alpha_score += 1
        Alpha_score_f = (Alpha_score/len(alpha_results))
        Alpha_score_p = Alpha_score_f * 100

        Beta_Score = 0
        for result in beta_results:
            if result == 'Beta':
                Beta_Score += 1
        Beta_Score_f = (Beta_Score/len(beta_results))
        Beta_Score_p = Beta_Score_f * 100

        Positron_score = 0
        for result in positron_results:
            if result == 'Positron':
                Positron_score += 1
        Positron_score_f = (Positron_score/len(positron_results))
        Positron_score_p = Positron_score_f * 100

        percentage_list = [Stable_score_p, Alpha_score_p, Beta_Score_p, Positron_score_p]
        average_p = mean(percentage_list)

        return (Stable_score, Stable_score_f, Stable_score_p, Alpha_score, Alpha_score_f, Alpha_score_p, Beta_Score,
                Beta_Score_f, Beta_Score_p, Positron_score, Positron_score_f, Positron_score_p, average_p)

    def tree_test():
        clf = tree.DecisionTreeClassifier()
        clf.fit(X, y)
        stable_results = clf.predict(stable_list_predict)
        alpha_results = clf.predict(alpha_list_predict)
        beta_results = clf.predict(beta_list_predict)
        positron_results = clf.predict(positron_list_predict)
        Stable_score = 0
        for result in stable_results:
            if result == 'Stable':
                Stable_score += 1

        Stable_score_f = (Stable_score / len(stable_results))
        Stable_score_p = Stable_score_f * 100

        Alpha_score = 0
        for result in alpha_results:
            if result == 'Alpha':
                Alpha_score += 1
        Alpha_score_f = (Alpha_score / len(alpha_results))
        Alpha_score_p = Alpha_score_f * 100

        Beta_Score = 0
        for result in beta_results:
            if result == 'Beta':
                Beta_Score += 1
        Beta_Score_f = (Beta_Score / len(beta_results))
        Beta_Score_p = Beta_Score_f * 100

        Positron_score = 0
        for result in positron_results:
            if result == 'Positron':
                Positron_score += 1
        Positron_score_f = (Positron_score / len(positron_results))
        Positron_score_p = Positron_score_f * 100

        percentage_list = [Stable_score_p, Alpha_score_p, Beta_Score_p, Positron_score_p]
        average_p = mean(percentage_list)

        return (Stable_score, Stable_score_f, Stable_score_p, Alpha_score, Alpha_score_f, Alpha_score_p, Beta_Score,
                Beta_Score_f, Beta_Score_p, Positron_score, Positron_score_f, Positron_score_p, average_p)

    def Native_Bayes_Test():
        clf = naive_bayes.GaussianNB()
        clf.fit(X, y)
        stable_results = clf.predict(stable_list_predict)
        alpha_results = clf.predict(alpha_list_predict)
        beta_results = clf.predict(beta_list_predict)
        positron_results = clf.predict(positron_list_predict)
        Stable_score = 0
        for result in stable_results:
            if result == 'Stable':
                Stable_score += 1

        Stable_score_f = (Stable_score / len(stable_results))
        Stable_score_p = Stable_score_f * 100

        Alpha_score = 0
        for result in alpha_results:
            if result == 'Alpha':
                Alpha_score += 1
        Alpha_score_f = (Alpha_score / len(alpha_results))
        Alpha_score_p = Alpha_score_f * 100

        Beta_Score = 0
        for result in beta_results:
            if result == 'Beta':
                Beta_Score += 1
        Beta_Score_f = (Beta_Score / len(beta_results))
        Beta_Score_p = Beta_Score_f * 100

        Positron_score = 0
        for result in positron_results:
            if result == 'Positron':
                Positron_score += 1
        Positron_score_f = (Positron_score / len(positron_results))
        Positron_score_p = Positron_score_f * 100

        percentage_list = [Stable_score_p, Alpha_score_p, Beta_Score_p, Positron_score_p]
        average_p = mean(percentage_list)
        return (Stable_score, Stable_score_f, Stable_score_p, Alpha_score, Alpha_score_f, Alpha_score_p, Beta_Score,
                Beta_Score_f, Beta_Score_p, Positron_score, Positron_score_f, Positron_score_p, average_p)

    Trial = '''

    SVC:
        Stable Score: %s
        Alpha Score: %s
        Beta Score: %s
        Positron Score: %s
        Average Score: %s

    Tree:
        Stable Score: %s
        Alpha Score: %s
        Beta Score: %s
        Positron Score: %s
        Average Score: %s

    Native Bayes:
        Stable Score: %s
        Alpha Score: %s
        Beta Score: %s
        Positron Score: %s
        Average Score: %s

        ''' % (svm_test()[2], svm_test()[5], svm_test()[8], svm_test()[11], svm_test()[12],
               tree_test()[2],tree_test()[5], tree_test()[8], tree_test()[11], tree_test()[12], Native_Bayes_Test()[2],
               Native_Bayes_Test()[5], Native_Bayes_Test()[8], Native_Bayes_Test()[11], Native_Bayes_Test()[12])

    f = open('data.txt', 'a')
    f.write('   ----------------------------')
    f.write('\n')
    f.write('\n')
    f.write(Trial)
    f.close()


if __name__ == '__main__':
    for number in range(20):
        ml_test()
