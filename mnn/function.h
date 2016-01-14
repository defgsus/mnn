/**	@file

	@brief basic functions

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_FUNCTION_H_INCLUDED
#define MNN_FUNCTION_H_INCLUDED

#include <cstddef>

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



struct DenseMatrix
{
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

    /** Propagates values from @p output into @p input */
    template <typename Float>
    static void bprop(
            Float* input, const Float* output, const Float* weight,
            size_t numIn, size_t numOut)
    {
        for (size_t i = 0; i < numIn; ++i, ++input)
        {
            Float sum = 0;
            Float* outp = output;
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
};







} // namespace MNN

#endif // MNN_FUNCTION_H_INCLUDED
