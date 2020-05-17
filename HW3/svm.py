import numpy as np
import cvxopt
import cvxopt.solvers



class LinearSVM():
    def __init__(self, c=10.0):
        self.c = c  
    
    def calc_alpha(self):
        x_len = len(self.train_data)
        y_len = len(self.train_label)

        symmetric_matrix = np.zeros(shape=(x_len, x_len))
        for i in range(x_len):
            for j in range(x_len):
                symmetric_matrix[i][j] = self.train_label[i] * self.train_label[j] * self.train_data[i].T.dot(self.train_data[j])

        q = cvxopt.matrix(symmetric_matrix)
        p = cvxopt.matrix(np.ones(x_len) * -1)
        a = cvxopt.matrix(np.array(self.train_label).astype('float'), (1, y_len))
        b = cvxopt.matrix(0.0)

        constrain_1 = np.identity(x_len) * -1
        constrain_2 = np.identity(x_len)
        g = cvxopt.matrix(np.vstack((constrain_1, constrain_2)))
        
        constrain_1 = np.zeros(x_len) * -1
        constrain_2 = np.ones(x_len) * self.c
        h = cvxopt.matrix(np.hstack((constrain_1, constrain_2)))
        
        cvxopt.solvers.options['show_progress'] = False
        self.alpha = cvxopt.solvers.qp(q, p, g, h, a, b)
        self.alpha = np.ravel(self.alpha['x'])
        self.alpha = np.expand_dims(self.alpha, axis=1)

    def calc_weight(self):
        self.weight = 0
        for i in self.support_vector_indices.tolist():
            self.weight += self.train_label[i] * self.alpha[i][0] * self.train_data[i].reshape((2, 1)) 

    def calc_bias(self):
        first_sv_index = self.support_vector_indices[0]
        self.bias = (1 / self.train_label[first_sv_index]) - self.weight.T.dot(self.train_data[first_sv_index].reshape((2, 1)))

    def calc_support_vector_indices(self):
        self.support_vector_indices = np.where(self.alpha > 10 ** -4)[0]

    def fit(self, train_data, train_label):
        if len(train_data) != len(train_label):
            raise ValueError("Length of training data and label must be same.")
        
        self.train_data = train_data
        self.train_label = train_label

        self.calc_alpha()
        self.calc_support_vector_indices()
        self.calc_weight()
        self.calc_bias()

    def judge_hyperplane_slop(self, w1, w2):
        if w2 == 0: raise ValueError("The weight 2 can't not be zero")
        if w1 == 0: return 'zero'

        y1 = (-w1 * 0 - self.bias) / w2
        y2 = (-w1 * 1 - self.bias) / w2

        if y2 - y1 > 0:
            return 'positive'
        else:
            return 'negative'

    def evaluate(self, test_data, test_label):
        w1, w2 = self.weight[0], self.weight[1]
        hyperplane_slop = self.judge_hyperplane_slop(w1, w2)

        judgement_right_count = 0
        for i in range(len(test_data)):
            data_x = test_data[i, 0]
            data_y = test_data[i, 1]
            hyperplane_height = -(w1 * data_x + self.bias) / w2
            
            dis = data_y - hyperplane_height

            if hyperplane_slop == 'positive':
                if dis > 0:
                    predict = 1 
                else:
                    predict = -1 
            elif hyperplane_slop == 'negative':
                if dis > 0:
                    predict = -1 
                else:
                    predict = 1 
            else:
                raise ValueError('This function does not support zero slop of hyperplane')

            if predict == 1 and predict == test_label[i]:
                judgement_right_count += 1
            elif predict == -1 and predict == test_label[i]:
                judgement_right_count += 1

        accuracy = (judgement_right_count / len(test_label)) * 100

        return accuracy

    def get_hyperplane_points(self, x_start, x_end):
        w1, w2 = self.weight[0], self.weight[1]
        selected_range = range(x_start, x_end + 1, 1)

        x = np.array(selected_range)
        y = list(map(lambda x: (-(w1 * x + self.bias) / w2).flatten(), selected_range))
        return x, y

class KernelSVM():
  def __init__(self, c=10):
    self.c = c
  
  def mapping(self, train_i, train_j):
    return 0

  def calc_alpha(self):
    x_len = len(self.train_data)
    y_len = len(self.train_label)

    symmetric_matrix = np.zeros(shape=(x_len, x_len))
    for i in range(x_len):
        for j in range(x_len):
            symmetric_matrix[i][j] = self.train_label[i] * self.train_label[j] * self.mapping(self.train_data[i].reshape((2, 1)), self.train_data[j].reshape((2, 1)))

    q = cvxopt.matrix(symmetric_matrix)
    p = cvxopt.matrix(np.ones(x_len) * -1)
    a = cvxopt.matrix(np.array(self.train_label).astype('float'), (1, y_len))
    b = cvxopt.matrix(0.0)

    constrain_1 = np.identity(x_len) * -1
    constrain_2 = np.identity(x_len)
    g = cvxopt.matrix(np.vstack((constrain_1, constrain_2)))
        
    constrain_1 = np.zeros(x_len) * -1
    constrain_2 = np.ones(x_len) * self.c
    h = cvxopt.matrix(np.hstack((constrain_1, constrain_2)))
        
    cvxopt.solvers.options['show_progress'] = False
    self.alpha = cvxopt.solvers.qp(q, p, g, h, a, b)
    self.alpha = np.ravel(self.alpha['x'])
    self.alpha = np.expand_dims(self.alpha, axis=1)
        
  def w_times_x(self, x):
    value = 0
    for i in self.support_vector_indices:
      value += self.train_label[i] * self.alpha[i][0] * self.mapping(self.train_data[i].reshape((2, 1)), x)
    
    return value

  def calc_bias(self):
    first_sv_index = self.support_vector_indices[0]
    self.bias = (1 / self.train_label[first_sv_index]) - self.w_times_x(self.train_data[first_sv_index])

  def calc_support_vector_indices(self):
    self.support_vector_indices = np.where(self.alpha > 10 ** -4)[0]

  def fit(self, train_data, train_label):
    if len(train_data) != len(train_label):
      raise ValueError("Length of training data and label must be same.")
    
    self.train_data = train_data
    self.train_label = train_label

    self.calc_alpha()
    self.calc_support_vector_indices()
    self.calc_bias()

  def evaluate(self, test_data, test_label):
    judgement_right_count = 0
    predictions = self.predict(test_data)

    for i in range(len(test_data)):
        if predictions[i] == 1 and test_label[i] == 1:
            judgement_right_count += 1
        else:
            judgement_right_count += 0

        if predictions[i] == -1 and test_label[i] == -1:
            judgement_right_count += 1
        else:
            judgement_right_count += 0

    accuracy = (judgement_right_count / len(test_label)) * 100
    return accuracy


  def predict(self, test_data):
    predictions = []
    for i in range(len(test_data)):
      prediction = self.w_times_x(test_data[i]) + self.bias
      prediction = 1 if prediction > 0 else -1
      predictions.append(prediction)
    
    return predictions
  
  def get_binary_class_area(self, x_start=3, x_end=8, y_start=0, y_end=5):
    positive_area = []
    negative_area = []

    for i in np.arange(x_start, x_end, 0.03):
      for j in np.arange(y_start, y_end, 0.03):
        prediction = self.predict(np.array([[i], [j]]).reshape((1, 2, 1)))[0]
        
        if prediction == 1:
          positive_area.append([i, j])
        else:
          negative_area.append([i, j])
    
    positive_area = np.array(positive_area)
    negative_area = np.array(negative_area)
    
    return positive_area, negative_area
  
class RBFSVM(KernelSVM):
  def __init__(self, c=10.0, sigma=5.0):
    super().__init__(c)
    self.sigma = sigma
  
  def mapping(self, x_i, x_j):
    x_i_minus_x_j = x_i - x_j
    return np.exp((x_i_minus_x_j).T.dot(x_i_minus_x_j) / (-2 * self.sigma ** 2))

class PolynomialSVM(KernelSVM):
  def __init__(self, c=10, p=2):
      super().__init__(c)
      self.p = p
  
  def mapping(self, x_i, x_j):
      return x_i.T.dot(x_j) ** self.p
