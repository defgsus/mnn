/**	@file

	@brief basic functions

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_FUNCTION_H_INCLUDED
#define MNN_FUNCTION_H_INCLUDED

#include <cstddef>
#include <cinttypes>
#include <cmath>

namespace MNN {


namespace Private {

    template <typename T>
    struct IsFloat { };

    template <>
    struct IsFloat<float> { typedef float Type; };

    template <>
    struct IsFloat<double> { typedef double Type; };

    template <>
    struct IsFloat<long double> { typedef long double Type; };

} // namespace Private




template <typename Float>
typename Private::IsFloat<Float>::Type rnd(Float min_, Float max_)
{
	return min_ + (Float)rand()/RAND_MAX * (max_ - min_);
}

template <typename Float>
typename Private::IsFloat<Float>::Type rndg(Float mean, Float dev)
{
	return mean + rnd((Float)0, (Float)1) * rnd(-dev,dev);
}


template <typename Float>
void apply_softmax(Float * states, size_t num)
{
    Float sum = Float(0);
    for (size_t i=0; i<num; ++i)
    {
        states[i] = std::exp(states[i]);
        sum += states[i];
    }
    if (sum != Float(0))
    for (size_t i=0; i<num; ++i)
        states[i] /= sum;
}


struct DenseMatrix
{
    /** Forward propagate @p input into @p output */
    template <typename Float, class Activation>
    static void fprop(
            const Float* input, Float* output, const Float* weight,
            size_t numIn, size_t numOut)
    {
        for (size_t o = 0; o < numOut; ++o, ++output)
        {
            Float sum = 0;
            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp, ++weight)
            {
                sum += *inp * *weight;
            }

            *output = Activation::activation(sum);
        }
    }

    /** Forward propagate @p input into @p output
        with bias for each output cell */
    template <typename Float, class Activation>
    static void fprop_bias(
            const Float* input, Float* output, const Float* bias,
            const Float* weight,
            size_t numIn, size_t numOut)
    {
        for (size_t o = 0; o < numOut; ++o, ++output, ++bias)
        {
            Float sum = *bias;
            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp, ++weight)
            {
                sum += *inp * *weight;
            }

            *output = Activation::activation(sum);
        }
    }

    /** Forward propagate @p input into @p output
        using transposed weight matrix. */
    template <typename Float, class Activation>
    static void fprop_transpose(
            const Float* input, Float* output, const Float* weight,
            size_t numIn, size_t numOut)
    {
        for (size_t o = 0; o < numOut; ++o, ++output)
        {
            Float sum = 0;
            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp)
            {
                sum += *inp * weight[i * numOut + o];
            }

            *output = Activation::activation(sum);
        }
    }

    /** Forward propagate @p input into @p output
        and scale before activation */
    template <typename Float, class Activation>
    static void fprop_scale(
            const Float* input, Float* output, const Float* weight,
            size_t numIn, size_t numOut, Float scale)
    {
        for (size_t o = 0; o < numOut; ++o, ++output)
        {
            Float sum = 0;
            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp, ++weight)
            {
                sum += *inp * *weight;
            }

            *output = Activation::activation(sum * scale);
        }
    }

    /** Propagates values from @p output into @p input */
    template <typename Float>
    static void bprop(
            Float* input, const Float* output, const Float* weight,
            size_t numIn, size_t numOut)
    {
        for (size_t i = 0; i < numIn; ++i, ++input)
        {
            Float sum = 0;
            const Float* outp = output;
            for (size_t o = 0; o < numOut; ++o, ++outp)
            {
                sum += *outp * weight[o * numIn + i];
            }
            *input = sum;
        }
    }

    /** Propagates values from @p output into @p input */
    template <typename Float>
    static void bprop_stride(
            Float* input, const Float* output, const Float* weight,
            size_t numIn, size_t numOut, size_t numInStride)
    {
        for (size_t i = 0; i < numIn; ++i, ++input)
        {
            Float sum = 0;
            const Float* outp = output;
            for (size_t o = 0; o < numOut; ++o, ++outp)
            {
                sum += *outp * weight[o * numInStride + i];
            }
            *input = sum;
        }
    }

    template <typename Float, class Activation>
    static void gradient_descent(
            const Float* input, const Float* output, const Float* error,
            Float* weight, Float* previous_delta,
            size_t numIn, size_t numOut, Float learn_rate, Float momentum)
    {
        for (auto o = 0; o < numOut; ++o, ++output, ++error)
        {
            Float de = Activation::derivative(*error, *output);

            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp, ++weight, ++previous_delta)
            {
                *previous_delta = momentum * *previous_delta
                                + learn_rate * de * *inp;
                *weight += *previous_delta;
            }
        }
    }


    template <typename Float, class Activation>
    static void gradient_descent_transpose(
            const Float* input, const Float* output, const Float* error,
            Float* weight, Float* previous_delta,
            size_t numIn, size_t numOut, Float learn_rate, Float momentum)
    {
        for (auto o = 0; o < numOut; ++o, ++output, ++error)
        {
            Float de = Activation::derivative(*error, *output);

            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp)
            {
                size_t idx = i * numOut + o;
                previous_delta[idx] = momentum * *previous_delta
                                + learn_rate * de * *inp;
                weight[idx] += previous_delta[idx];
            }
        }
    }

    template <typename Float, class Activation>
    static void gradient_descent_bias(
            const Float* output, const Float* error,
            Float* bias,
            size_t numOut, Float learn_rate)
    {
        for (auto o = 0; o < numOut; ++o, ++output, ++error, ++bias)
        {
            Float de = Activation::derivative(*error, *output);

            *bias += learn_rate * de;
        }
    }

};




struct DenseMatrixDropout
{
    /** Forward propagate @p input into @p output with drop-out.
        @p drop_input contains 0 (use cell) or 1 (ignore cell) for each input cell. */
    template <typename Float, typename Bool, class Activation>
    static void fprop(
            const Float* input, Float* output, const Float* weight,
            const Bool* drop_input,
            size_t numIn, size_t numOut)
    {
        for (size_t o = 0; o < numOut; ++o, ++output)
        {
            Float sum = 0;
            const Float* inp = input;
            const Bool* dropInp = drop_input;
            for (auto i = 0; i < numIn; ++i, ++inp, ++dropInp, ++weight)
            {
                if (!*dropInp)
                    sum += *inp * *weight;
            }

            *output = Activation::activation(sum);
        }
    }


    /** Propagates values from @p output into @p input.
        @p drop_input contains 0 (use cell) or 1 (ignore cell) for each input cell.
        Dropped input cells are set to zero. */
    template <typename Float, typename Bool>
    static void bprop(
            Float* input, const Float* output, const Float* weight,
            const Bool* drop_input,
            size_t numIn, size_t numOut)
    {
        for (size_t i = 0; i < numIn; ++i, ++input)
        {
            if (drop_input[i])
            {
                *input = Float(0);
                continue;
            }
            Float sum = 0;
            Float* outp = output;
            for (size_t o = 0; o < numOut; ++o, ++outp)
            {
                sum += *outp * weight[o * numIn + i];
            }
            *input = sum;
        }
    }

    /** Propagates values from @p output into @p input.
        @p drop_input contains 0 (use cell) or 1 (ignore cell) for each input cell.
        Dropped input cells are set to zero. */
    template <typename Float, typename Bool>
    static void bprop_stride(
            Float* input, const Float* output, const Float* weight,
            const Bool* drop_input,
            size_t numIn, size_t numOut, size_t numInStride)
    {
        for (size_t i = 0; i < numIn; ++i, ++input)
        {
            if (drop_input[i])
            {
                *input = Float(0);
                continue;
            }
            Float sum = 0;
            const Float* outp = output;
            for (size_t o = 0; o < numOut; ++o, ++outp)
            {
                sum += *outp * weight[o * numInStride + i];
            }
            *input = sum;
        }
    }

    template <typename Float, typename Bool, class Activation>
    static void gradient_descent(
            const Float* input, const Float* output, const Float* error,
            Float* weight, Float* previous_delta,
            const Bool* drop_input,
            size_t numIn, size_t numOut, Float learn_rate, Float momentum)
    {
        for (auto o = 0; o < numOut; ++o, ++output, ++error)
        {
            Float de = Activation::derivative(*error, *output);

            const Float* inp = input;
            for (auto i = 0; i < numIn; ++i, ++inp, ++weight, ++previous_delta)
            {
                if (drop_input[i])
                    continue;
                *previous_delta = momentum * *previous_delta
                                + learn_rate * de * *inp;
                *weight += *previous_delta;
            }
        }
    }
};



/** Convolution functions.
    All arrays are row-major. */
struct ConvolutionMatrix
{
    /** Propagate @p input into @p output, using the weights in @p kernel.
        @param input has size @p inputWidth * @p inputHeight
        @param output has the size ((@p inputWidth - @p kernelWidth) / @p strideX + 1)
                                 * ((@p inputHeight - @p kernelHeight) / @p strideY + 1)
        @param kernel has size @p kernelWidth * @p kernelHeight
    */
    template <typename Float, class Act>
    static void fprop(const Float* input, Float* output, const Float* kernel,
                      size_t inputWidth, size_t inputHeight,
                      size_t kernelWidth, size_t kernelHeight,
                      size_t strideX = 1, size_t strideY = 1)
    {
        const size_t
                scanWidth = (inputWidth - kernelWidth + 1),
                scanHeight = (inputHeight - kernelHeight + 1);
        for (size_t sy = 0; sy < scanHeight; sy += strideY)
        for (size_t sx = 0; sx < scanWidth; sx += strideX, ++output)
        {
            const Float* w = kernel;
            Float sum = 0;
            for (size_t iy = 0; iy < kernelHeight; ++iy)
            {
                const Float* inp = &input[(sy + iy) * inputWidth + sx];
                for (size_t ix = 0; ix < kernelWidth; ++ix, ++w, ++inp)
                    sum += *w * *inp;
            }
            *output = Act::activation(sum);
        }
    }

    /** Back-propagate @p output into @p input, using the weights in @p kernel.
        @param input has size @p inputWidth * @p inputHeight
        @param output has the size ((@p inputWidth - @p kernelWidth) / @p strideX + 1)
                                 * ((@p inputHeight - @p kernelHeight) / @p strideY + 1)
        @param kernel has size @p kernelWidth * @p kernelHeight
    */
    template <typename Float>
    static void bprop(Float* input, const Float* output, const Float* kernel,
                      size_t inputWidth, size_t inputHeight,
                      size_t kernelWidth, size_t kernelHeight,
                      size_t strideX = 1, size_t strideY = 1)
    {
        const size_t
                scanWidth = (inputWidth - kernelWidth + 1),
                scanHeight = (inputHeight - kernelHeight + 1),
                inputSize = inputWidth * inputHeight;
        // first clear input
        for (size_t i = 0; i < inputSize; ++i)
            input[i] = Float(0);
        // then sum into input
        for (size_t sy = 0; sy < scanHeight; sy += strideY)
        for (size_t sx = 0; sx < scanWidth; sx += strideX, ++output)
        {
            const Float* w = kernel;
            for (size_t iy = 0; iy < kernelHeight; ++iy)
            {
                Float* inp = &input[(sy + iy) * inputWidth + sx];
                for (size_t ix = 0; ix < kernelWidth; ++ix, ++w, ++inp)
                    *inp += *w * *output;
            }
        }
    }

    /** Gradient descent on convolution filter.
        @param input is the @p inputWidth * @p inputHeight input map
        @param errorDerivative is the partial derivative of the produced error
               with size ((@p inputWidth - @p kernelWidth) / @p strideX + 1)
                       * ((@p inputHeight - @p kernelHeight) / @p strideY + 1)
        @param kernel is the weight matrix of size @p kernelWidth * @p kernelHeight
        @param previousDelta has the same size as @p kernel and stores the weight deltas
               for the momentum
        @param kernelScratch has the same size as @p kernel and is used to accumulate
               the weight deltas before application.
    */
    template <typename Float>
    static void gradient_descent(
                const Float* input, const Float* errorDerivative,
                Float* kernel, Float* previousDelta, Float* kernelScratch,
                size_t inputWidth, size_t inputHeight,
                size_t kernelWidth, size_t kernelHeight,
                size_t strideX, size_t strideY,
                Float learnRate, Float momentum)
    {
        const size_t
                scanWidth = (inputWidth - kernelWidth + 1),
                scanHeight = (inputHeight - kernelHeight + 1),
                kernelSize = kernelWidth * kernelHeight;

        // clear scratch buffer
        for (size_t i=0; i<kernelSize; ++i)
            kernelScratch[i] = Float(0);

        // accumulate weight deltas into scratch buffer
        for (size_t sy = 0; sy < scanHeight; sy += strideY)
        for (size_t sx = 0; sx < scanWidth; sx += strideX, ++errorDerivative)
        {
            Float *w = kernelScratch;
            for (size_t iy = 0; iy < kernelHeight; ++iy)
            {
                const Float* inp = &input[(sy + iy) * inputWidth + sx];
                for (size_t ix = 0; ix < kernelWidth; ++ix, ++w, ++inp)
                {
                    *w += *errorDerivative * *inp;
                }
            }
        }

        // adjust weights and momentum
        for (size_t i=0; i<kernelSize; ++i, ++kernel, ++kernelScratch, ++previousDelta)
        {
            *previousDelta =
                      momentum * *previousDelta
                    + learnRate * *kernelScratch;
            *kernel += *previousDelta;
        }
    }

};






} // namespace MNN

#endif // MNN_FUNCTION_H_INCLUDED
