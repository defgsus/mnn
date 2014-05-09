/**	@file

	@brief Perceptron implementation

	@author def.gsus-
	@version 2012/10/15 started
*/

#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_PERCEPTRON Perceptron<Float, ActFunc>

MNN_TEMPLATE
MNN_PERCEPTRON::Perceptron(size_t nrIn, size_t nrOut, Float learnRate)
	:	learnRate_	(learnRate)
{
	resize(nrIn, nrOut);
}

MNN_TEMPLATE
MNN_PERCEPTRON::~Perceptron()
{

}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_PERCEPTRON::resize(size_t nrIn, size_t nrOut)
{
	input_.resize(nrIn);
	output_.resize(nrOut);
	weight_.resize(nrIn * nrOut);
}

MNN_TEMPLATE
size_t MNN_PERCEPTRON::nrIn() const
{
	return input_.size();
}

MNN_TEMPLATE
size_t MNN_PERCEPTRON::nrOut() const
{
	return output_.size();
}


MNN_TEMPLATE
void MNN_PERCEPTRON::brainwash()
{
	// reset in/out
	for (auto e = input_.begin(); e != input_.end(); ++e)
		*e = 0.0;
	for (auto e = output_.begin(); e != output_.end(); ++e)
		*e = 0.0;

	if (input_.empty() || output_.empty()) return;

	// randomize weights (assume normalized states)
	Float f = 1.0 / sqrt(input_.size());
	for (auto e = weight_.begin(); e != weight_.end(); ++e)
		*e = rnd(-f, f);
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_PERCEPTRON::fprop(Float * input, Float * output)
{
	// copy to internal data
	for (auto i = input_.begin(); i != input_.end(); ++i, ++input)
		*i = *input;

	// propagate
	auto w = weight_.begin();
	for (auto o = output_.begin(); o != output_.end(); ++o)
	{
		Float sum = 0;
		for (auto i = input_.begin(); i != input_.end(); ++i, ++w)
		{
			sum += *i * *w;
		}

		*o = ActFunc::activation(sum);
	}

	// copy to caller
	std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_PERCEPTRON::bprop(Float * error, Float * error_output, Float global_learn_rate)
{
	global_learn_rate *= learnRate_;

	Float * e;

	// pass error through
	if (error_output)
	for (size_t i = 0; i<input_.size(); ++i, ++error_output)
	{
		Float sum = 0;
		e = error;
		for (size_t o = 0; o < output_.size(); ++o, ++e)
		{
			sum += *e * weight_[o * input_.size() + i];
		}
		*error_output = sum;
	}

	// backprob derivative
	auto w = weight_.begin();
	e = error;
	for (auto o = output_.begin(); o != output_.end(); ++o, ++e)
	{
		Float de = ActFunc::derivative((Float)*e, *o);

		for (auto i = input_.begin(); i != input_.end(); ++i, ++w)
		{
			*w += learnRate_ * de * *i;
		}

	}
}


// ----------- info -----------------------

MNN_TEMPLATE
void MNN_PERCEPTRON::info(std::ostream &out) const
{
	out <<   "name       : " << name()
		<< "\ninputs     : " << input_.size()
		<< "\noutputs    : " << output_.size()
		<< "\nactivation : " << ActFunc::static_name()
		<< "\n";
}

MNN_TEMPLATE
void MNN_PERCEPTRON::dump(std::ostream &out) const
{
	out << "inputs:";
	for (auto e = input_.begin(); e != input_.end(); ++e)
		out << " " << *e;

	out << "\noutputs:";
	for (auto e = output_.begin(); e != output_.end(); ++e)
		out << " " << *e;

	out << "\nweights:";
	for (auto e = weight_.begin(); e != weight_.end(); ++e)
		out << " " << *e;

	out << "\n";
}






#undef MNN_TEMPLATE
#undef MNN_PERCEPTRON
