/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/

#ifndef MNNSRC_FACTORY_H
#define MNNSRC_FACTORY_H

#include <fstream>

#include "exception.h"
#include "layer.h"
#include "activation.h"
#include "feedforward.h"
#include "convolution.h"
#include "rbm.h"
#include "stack_parallel.h"
#include "stack_serial.h"

namespace MNN {

template <typename Float>
class Factory
{
public:

    static Layer<Float>* createLayer(const std::string& id, const std::string& activation_id);

    template <class ActFunc>
    static Layer<Float>* createLayer(const std::string& id);

    /** Loads a complete network or single layer.
        @throws MNN::Exception */
    static Layer<Float>* loadTextFile(const std::string& fn);

private:

    static Layer<Float>* createFromStream_(std::istream&);
};

#include "factory_impl.inl"

} // namespace MNN

#endif // MNNSRC_FACTORY_H

