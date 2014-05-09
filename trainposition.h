/** @file trainposition.h

    @brief

    <p>(c) 2014, stefan.berke@modular-audio-graphics.com</p>

    <p>created 5/10/2014</p>
*/

#ifndef TRAINPOSITION_H
#define TRAINPOSITION_H

#include <vector>
#include <QString>
#include <mnn/layer.h>

class TrainPosition
{
public:
    TrainPosition();

    void exec();
    void test();

    bool loadFile(const QString filename);

private:

    MNN::Layer<float> * net_;

    static const int entryLength_ = 1 + 62*2;
    std::vector<float> data_;
    int num_;
};

#endif // TRAINPOSITION_H
