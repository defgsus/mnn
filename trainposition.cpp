/** @file trainposition.cpp

    @brief

    <p>(c) 2014, stefan.berke@modular-audio-graphics.com</p>

    <p>created 5/10/2014</p>
*/

#include <iomanip>

#include <QFile>
#include <QDataStream>

#include "trainposition.h"
#include "mnn/mnn.h"

TrainPosition::TrainPosition()
{
    float lr = 0.006;

    auto net = new MNN::StackSerial<float>;

    net->addLayer( new MNN::FeedForward<float, MNN::Activation::Logistic>(entryLength_-1,500, 1*lr) );
    net->addLayer( new MNN::FeedForward<float, MNN::Activation::Logistic>(500,100, 1*lr) );
    net->addLayer( new MNN::FeedForward<float, MNN::Activation::Linear>(100,1, 1*lr) );

    net_ = net;

    net->info();
}

bool TrainPosition::loadFile(const QString filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
        return false;

    data_.clear();
    num_ = 0;

    float f;
    QDataStream in(&file);
    while (!in.atEnd())
    {
        in >> f;
        data_.push_back(f);
    }
    num_ = data_.size() / entryLength_;

    return true;
}


void TrainPosition::exec()
{
    loadFile("/home/defgsus/prog/shatra/positions.nn_input");

    std::cout << "training "<<num_<<" shatra positions...\n";

    net_->brainwash();

    float output, error = 0.1, e;
    size_t count = 0;

    while (error > 0.01 && count<100000)
    {
        // choose pattern
        size_t nr = rand() % num_;
        // feed forward
        net_->fprop(&data_[nr * entryLength_ + 1], &output);
        // compare
        e = data_[nr * entryLength_] - output;
        // average error over time
        error += (fabs(e) - error) / 1000.f;
        // train
        net_->bprop(&e);
        if (count % 5 == 0)
            std::cout << "av. error " << error << " out " << output << "           \r";
        //net.print();
        count++;
    }
    std::cout << "\ntook "<<count<<" epochs                                      \n";
}

void TrainPosition::test()
{
    float output, error = 1.0, e;

    std::cout << "testing...\n";
    error = 0.0;
    for (int i=0; i<num_;i++)
    {
        net_->fprop(&data_[i * entryLength_ + 1], &output);
        // compare
        e = fabs(data_[i * entryLength_] - output);
        error += e;
        std::cout << "pos " << std::setw(8) << i << " = " << std::setw(10) << output << " (abs. error " << e << ")\n";
    }
    error /= num_;
    std::cout << "average abs. error " << error << "\n";
}
