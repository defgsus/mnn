/**	@file

	@brief activation function objects

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_ACTIVATION_H_INCLUDED
#define MNN_ACTIVATION_H_INCLUDED

namespace MNN {

namespace Activation {

/** base class for activation function objects */
struct Base
{
	virtual const char * name() const = 0;
};


/** linear activation */
struct Linear : public Base
{
	static const char * static_name() { return "linear"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return in; }

	template <typename Float>
	static Float derivative(Float error, Float /*state*/) { return error; }
};


/** tangens hyperbolicus activation */
struct Tanh : public Base
{
	static const char * static_name() { return "tangens hyperbolicus"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return tanh(in); }

	template <typename Float>
	static Float derivative(Float error, Float state) { return (1.0 - state * state) * error; }
};

/** classical logistic activation */
struct Logistic : public Base
{
	static const char * static_name() { return "logistic"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return 1.0/(1.0 + exp(-in)); }

	template <typename Float>
	static Float derivative(Float error, Float state) { return state * (1.0 - state) * error; }
};

/** logistic10 activation */
struct Logistic10 : public Base
{
	static const char * static_name() { return "logistic10"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return 1.0/(1.0 + exp(10.0 * -in)); }

	template <typename Float>
	static Float derivative(Float error, Float state) { return state * (1.0 - state) * error; }
};


/** cosine activation */
struct Cosine : public Base
{
	static const char * static_name() { return "cosine"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return cos(in); }

	template <typename Float>
	static Float derivative(Float error, Float state) { return -sin(state) * error; }
};

/** smooth activation */
struct Smooth : public Base
{
	static const char * static_name() { return "smooth"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return in*in*(3.0-2.0*in); }

	template <typename Float>
	static Float derivative(Float error, Float state) { return 6.0*(1.0-state) * error; }
};

/** smooth2 activation */
struct Smooth2 : public Base
{
	static const char * static_name() { return "smooth2"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return in*in*in*(6.0*in*in - 15.0*in + 10.0); }

	template <typename Float>
	static Float derivative(Float error, Float state) { return 30.0 * pow(state-1.0, 2.0) * state * state * error; }
};

} // namespace Activation

} // namespace MNN





#endif // MNN_ACTIVATION_H_INCLUDED
