/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#ifndef TRAINMNIST_H
#define TRAINMNIST_H


class TrainMnist
{
public:
    TrainMnist();
    ~TrainMnist();

    void exec();

private:
    struct Private;
    Private * p_;
};

#endif // TRAINMNIST_H
