/**	@file

	@brief basic functions

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_FUNCTION_H_INCLUDED
#define MNN_FUNCTION_H_INCLUDED

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











} // namespace MNN

#endif // MNN_FUNCTION_H_INCLUDED
