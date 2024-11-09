#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

// headers
// mnist loader
class MnistLoader {
  int width;
  int height;

 public:
  int get_width() const { return width; }
  int get_height() const { return height; }
  vector<vector<double> > read_images(const string&, const bool);
  vector<vector<double> > read_labels(const string&);
  void display_image(const vector<double>&, const vector<double>&, const int,
                     const int);
};

// layer
class Layer {
  int input_size;
  int output_size;
  double learning_rate;

 public:
  Layer(const int is, const int os) : input_size(is), output_size(os) {};
  virtual ~Layer() {}

  int get_input_size() const { return input_size; }
  int get_output_size() const { return output_size; }
  double get_learning_rate() const { return learning_rate; }
  void set_learning_rate(const double lr) { learning_rate = lr; }

  virtual vector<double> forward(const vector<double>&) = 0;
  virtual vector<double> backward(const vector<double>&) = 0;
};

// linear layer
class LinearLayer : public Layer {
  vector<double> input;

  vector<vector<double> > weights;
  vector<double> bias;

 public:
  LinearLayer(const int is, const int os) : Layer(is, os) {
    const int input_size = get_input_size();
    const int output_size = get_output_size();

    input = vector<double>(input_size,0);

    weights = vector<vector<double> >(input_size, vector<double>(output_size));
    bias = vector<double>(output_size, 0);

    for (int i = 0; i < input_size; i++) {
      for (int j = 0; j < output_size; j++) {
        weights[i][j] = 0.01 * (double)rand() / RAND_MAX;
      }
    }
  };
  ~LinearLayer() {}

  vector<double> forward(const vector<double>&);
  vector<double> backward(const vector<double>&);
};

// relu layer
class ReLULayer : public Layer {
  vector<double> mask;

 public:
  ReLULayer(const int size) : Layer(size, size) {
    mask = vector<double>(size, 0);
  }
  ~ReLULayer() {}

  vector<double> forward(const vector<double>&);
  vector<double> backward(const vector<double>&);
};

// sigmoid layer
class SigmoidLayer : public Layer {
  vector<double> output;

 public:
  SigmoidLayer(const int size) : Layer(size, size) {
    output = vector<double>(size, 0);
  }
  ~SigmoidLayer() {}

  vector<double> forward(const vector<double>&);
  vector<double> backward(const double);
};

// loss layer
class LossLayer {
  vector<double> y;
  vector<double> t;

 public:
  LossLayer(const int size) {
    y = vector<double>(size,0);
    t = vector<double>(size,0);
  }
  double forward(const vector<double>&, const vector<double>&);
  vector<double> backward();
};

// neural network
class NeuralNetwork {
  vector<Layer*> layers;
  LossLayer* loss_layer;
  double learning_rate;

 public:
  NeuralNetwork(const double lr) : learning_rate(lr) {}
  ~NeuralNetwork() {
    for (int i = 0; i < layers.size(); i++) {
      delete layers[i];
    }
  }
  void add_layer(Layer* layer) {
    layer->set_learning_rate(learning_rate);
    layers.push_back(layer);
  }
  void add_loss_layer(LossLayer* ll) { loss_layer = ll; }

  double calculate_loss(const vector<double>&, const vector<double>&);
  int calculate_accuracy(const vector<double>&, const vector<double>&);

  void train(const vector<vector<double> >&, const vector<vector<double> >&,
             const int);
  void test(const vector<vector<double> >&, const vector<vector<double> >&);
  vector<double> predict(const vector<double>&);
};

// methods
// mnist loader
int reverse_int(int i) {
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<vector<double> > MnistLoader::read_images(const string& path,
                                                 const bool is_train) {
  ifstream ifs(path);
  if (!ifs.is_open()) {
    cout << path << " cannot be opened." << endl;
    return vector<vector<double> >(0);
  }

  int magic_number = 0;
  int number_of_images = 0;
  int number_of_rows = 0;
  int number_of_columns = 0;

  ifs.read((char*)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  ifs.read((char*)&number_of_images, sizeof(number_of_images));
  number_of_images = reverse_int(number_of_images);
  ifs.read((char*)&number_of_rows, sizeof(number_of_rows));
  number_of_rows = reverse_int(number_of_rows);
  ifs.read((char*)&number_of_columns, sizeof(number_of_columns));
  number_of_columns = reverse_int(number_of_columns);

  cout << number_of_images << " images found, each with dimensions "
       << number_of_columns << "x" << number_of_rows << "." << endl;

  if (is_train) {
    width = number_of_columns;
    height = number_of_rows;
  }

  vector<vector<double> > images(number_of_images);

  for (int i = 0; i < number_of_images; i++) {
    images[i].assign(number_of_rows * number_of_columns, 0);

    for (int j = 0; j < number_of_rows * number_of_columns; j++) {
      unsigned char temp = 0;
      ifs.read((char*)&temp, sizeof(temp));
      images[i][j] = ((double)temp) / 255.0;
    }
  }

  return images;
}

vector<vector<double> > MnistLoader::read_labels(const string& path) {
  ifstream ifs(path);
  if (!ifs.is_open()) {
    cout << path << " cannot be opened." << endl;
    return vector<vector<double> >(0);
  }

  int magic_number = 0;
  int number_of_labels = 0;

  ifs.read((char*)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  ifs.read((char*)&number_of_labels, sizeof(number_of_labels));
  number_of_labels = reverse_int(number_of_labels);

  cout << number_of_labels << " labels found." << endl;

  vector<vector<double> > labels(number_of_labels);

  for (int i = 0; i < number_of_labels; i++) {
    labels[i].assign(10, 0);

    unsigned char temp = 0;
    ifs.read((char*)&temp, sizeof(temp));
    labels[i][(int)temp] = 1.0;
  }

  return labels;
}

void MnistLoader::display_image(const vector<double>& image,
                                const vector<double>& label, const int width,
                                const int height) {
  for (int i = 0; i < 10; i++) {
    if (label[i] > 0.5) {
      cout << "label: " << i << endl;
      break;
    }
  }

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      double pixel = image[i * width + j];

      if (pixel > 0.75) {
        cout << "░░";
      } else if (pixel > 0.50) {
        cout << "▒▒";
      } else if (pixel > 0.25) {
        cout << "▓▓";
      } else {
        cout << "██";
      }
    }
    cout << endl;
  }
}

// layer
// linear layer
vector<double> LinearLayer::forward(const vector<double>& x) {
  const int input_size = get_input_size();
  const int output_size = get_output_size();

  vector<double> y(output_size, 0);

  copy(x.begin(), x.end(), input.begin());

  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < input_size; j++) {
      y[i] += input[j] * weights[j][i];
    }
    y[i] += bias[i];
  }

  return y;
}

vector<double> LinearLayer::backward(const vector<double>& dout) {
  const int input_size = get_input_size();
  const int output_size = get_output_size();

  const double learning_rate = get_learning_rate();

  vector<double> dx(input_size, 0);

  for (int i = 0; i < input_size; i++) {
    for (int j = 0; j < output_size; j++) {
      dx[i] += dout[j] * weights[i][j];
    }
  }

  for (int i = 0; i < input_size; i++) {
    for (int j = 0; j < output_size; j++) {
      weights[i][j] -= learning_rate * input[i] * dout[j];
    }
  }

  for (int i = 0; i < output_size; i++) {
    bias[i] -= learning_rate * dout[i];
  }

  return dx;
}

// relu layer
vector<double> ReLULayer::forward(const vector<double>& x) {
  const int size = get_input_size();

  vector<double> y(size, 0);

  for (int i = 0; i < size; i++) {
    if (x[i] > 0) {
      mask[i] = 1;
    } else {
      mask[i] = 0;
    }
  }

  for (int i = 0; i < size; i++) {
    y[i] = x[i] * mask[i];
  }

  return y;
}

vector<double> ReLULayer::backward(const vector<double>& dout) {
  const int size = get_input_size();

  vector<double> dx(size, 0);

  for (int i = 0; i < size; i++) {
    dx[i] = dout[i] * mask[i];
  }

  return dx;
}

// sigmoid layer
vector<double> SigmoidLayer::forward(const vector<double>& x) {
  const int size = get_input_size();

  for (int i = 0; i < size; i++) {
    cout << i << " " << output.size() << endl;
    output[i] = 1.0 / (1.0 + exp(-x[i]));
  }

  return output;
}

vector<double> SigmoidLayer::backward(double dout) {
  const int size = get_input_size();

  vector<double> dx(size, 0);

  for (int i = 0; i < size; i++) {
    dx[i] = dout * (1.0 - output[i]) * output[i];
  }

  return dx;
}

// loss layer
double LossLayer::forward(const vector<double>& _x, const vector<double>& _t) {
  copy(_x.begin(), _x.end(), y.begin());
  copy(_t.begin(), _t.end(), t.begin());

  const int size = y.size();

  double loss = 0;

  for (int i = 0; i < size; i++) {
    loss += (y[i] - t[i]) * (y[i] - t[i]) / 2;
  }

  return loss;
}

vector<double> LossLayer::backward() {
  const int size = y.size();

  vector<double> dx(size, 0);

  for (int i = 0; i < size; i++) {
    dx[i] = (y[i] - t[i]);
  }

  return dx;
}

// neural network
vector<double> NeuralNetwork::predict(const vector<double>& x) {
  vector<double> y = vector<double>(x.size());

  copy(x.begin(), x.end(), y.begin());

  for (int i = 0; i < layers.size(); i++) {
    y = layers[i]->forward(y);
  }

  return y;
}

double NeuralNetwork::calculate_loss(const vector<double>& x,
                                     const vector<double>& t) {
  const vector<double> y = predict(x);
  return loss_layer->forward(y, t);
}

int NeuralNetwork::calculate_accuracy(const vector<double>& x,
                                      const vector<double>& t) {
  const vector<double> y = predict(x);

  int predicted_label = 0;
  int label = 0;
  double confidence = -numeric_limits<double>::infinity();

  for (int i = 0; i < y.size(); i++) {
    if (y[i] > confidence) {
      predicted_label = i;
      confidence = y[i];
    }
    if (t[i] > 0.5) {
      label = i;
    }
  }

  if (predicted_label == label) {
    return 1;
  }
  return 0;
}

void NeuralNetwork::train(const vector<vector<double> >& images,
                          const vector<vector<double> >& labels,
                          const int epochs) {
  const int image_size = images.size();
  const int layer_size = layers.size();
  const int label_size = labels[0].size();

  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_error = 0;
    int number_of_corrections = 0;

    for (int i = 0; i < image_size; i++) {
      const vector<double> x = images[i];
      const vector<double> t = labels[i];

      total_error += calculate_loss(x, t);
      if (i%100==0){
      number_of_corrections += calculate_accuracy(x,t);
      }

      vector<double> dout = loss_layer->backward();

      for (int i = layers.size() - 1; i >= 0; i--) {
        dout = layers[i]->backward(dout);
      }
    }

    cout << "Epoch: " << epoch + 1 << " / " << epochs
         << "; Error: " << total_error
         << "; Accuracy: " << ((float)number_of_corrections) / image_size * 10000
         << " %" << endl;
  }
}

void NeuralNetwork::test(const vector<vector<double> >& images,
                         const vector<vector<double> >& labels) {
  const int image_size = images.size();
  const int label_size = labels[0].size();

  int number_of_corrections = 0;

  for (int i = 0; i < image_size; i++) {
    vector<double> y = predict(images[i]);

    int label = 0;
    int predicted_label = 0;
    double confidence = -numeric_limits<double>::infinity();

    for (int j = 0; j < label_size; j++) {
      if (labels[i][j] > 0.5) {
        label = j;
      }

      const double _confidence = y[j];

      if (_confidence > confidence) {
        predicted_label = j;
        confidence = _confidence;
      }
    }

    if (label == predicted_label) {
      number_of_corrections++;
    }

    if (i < 10) {
      cout << "Predicted label: " << predicted_label << endl;
      cout << "Actual ";
      (new MnistLoader())->display_image(images[i], labels[i], 28, 28);
    }
  }

  cout << "Test accuracy: " << ((float)number_of_corrections) / image_size * 100
       << " %" << endl;
}

int main() {
  // initialize
  srand(time(NULL));

  // load data
  MnistLoader mnist_loader;
  vector<vector<double> > train_images, test_images;
  vector<vector<double> > train_labels, test_labels;

  train_images = mnist_loader.read_images("./train-images-idx3-ubyte", true);
  test_images = mnist_loader.read_images("./t10k-images-idx3-ubyte", false);
  train_labels = mnist_loader.read_labels("./train-labels-idx1-ubyte");
  test_labels = mnist_loader.read_labels("./t10k-labels-idx1-ubyte");

  int display_index = rand() % train_images.size();

  mnist_loader.display_image(
      train_images[display_index], train_labels[display_index],
      mnist_loader.get_width(), mnist_loader.get_height());

  // build the model
  NeuralNetwork nn = NeuralNetwork(1e-2);

  Layer* layer1 =
      new LinearLayer(mnist_loader.get_width() * mnist_loader.get_height(), 16);
  Layer* layer2 = new ReLULayer(16);
  Layer* layer3 = new LinearLayer(16, 10);
  Layer* layer4 = new ReLULayer(10);
  Layer* layer5 = new LinearLayer(10, train_labels[0].size());
  LossLayer* loss_layer = new LossLayer(10);

  nn.add_layer(layer1);
  nn.add_layer(layer2);
  nn.add_layer(layer3);
  nn.add_layer(layer4);
  nn.add_layer(layer5);
  nn.add_loss_layer(loss_layer);

  // train
  nn.train(train_images, train_labels, 10);

  // test
  nn.test(test_images, test_labels);
}
