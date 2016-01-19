/**	@file

	@brief MNN wrapper header

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNNSRC_MNN_H_INCLUDED
#define MNNSRC_MNN_H_INCLUDED

#include "mnn/activation.h"
#include "mnn/function.h"
#include "mnn/interface.h"
#include "mnn/layer.h"
#include "mnn/stack_serial.h"
#include "mnn/stack_parallel.h"
#include "mnn/feedforward.h"
#include "mnn/convolution.h"
#include "mnn/rbm.h"

namespace MNN {


/** Sets number of input and output cells to @p numInOut
    and sets all weights to zero except the weight connecting
    the same input and output cell. */
template <class Layer>
void initPassThrough(Layer* layer, typename Layer::type amplitude = 1, size_t numInOut = 0)
{
    if (numInOut != 0)
        layer->resize(numInOut, numInOut);
    layer->brainwash();
    for (size_t o=0; o<layer->numOut(); ++o)
    for (size_t i=0; i<layer->numIn(); ++i)
        layer->setWeight(i, o, (i == o) * amplitude);
}



} // namespace MNN

#endif // MNNSRC_MNN_H_INCLUDED
