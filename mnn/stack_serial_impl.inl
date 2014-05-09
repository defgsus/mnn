/**	@file

	@brief Serial Stack implementation

	@author def.gsus-
	@version 2012/10/19 started
*/

#define MNN_TEMPLATE template <typename Float>
#define MNN_STACKSERIAL StackSerial<Float>

MNN_TEMPLATE
MNN_STACKSERIAL::StackSerial()
{

}

MNN_TEMPLATE
MNN_STACKSERIAL::~StackSerial()
{
	// free layers
	for (auto l = layer_.begin(); l != layer_.end(); ++l)
		delete *l;
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_STACKSERIAL::resize(size_t nrIn, size_t nrOut)
{
	if (layer_.empty()) return;

	layer_[0]->resize(nrIn, layer_[0]->nrOut());
	layer_[layer_.size()-1]->resize(layer_[layer_.size()-1]->nrIn(), nrOut);

	resizeBuffers_();
}

MNN_TEMPLATE
size_t MNN_STACKSERIAL::nrIn() const
{
	return (layer_.empty())? 0 : layer_[0]->nrIn();
}

MNN_TEMPLATE
size_t MNN_STACKSERIAL::nrOut() const
{
	return (layer_.empty())? 0 : layer_[layer_.size()-1]->nrIn();
}


MNN_TEMPLATE
void MNN_STACKSERIAL::brainwash()
{
	for (auto l = layer_.begin(); l != layer_.end(); ++l)
		(*l)->brainwash();
}

// ----------- layer interface -----------

MNN_TEMPLATE
size_t MNN_STACKSERIAL::nrLayer() const
{
	return layer_.size();
}

MNN_TEMPLATE
void MNN_STACKSERIAL::add(Layer<Float> * layer)
{
	layer_.push_back(layer);
	resizeBuffers_();
}

MNN_TEMPLATE
void MNN_STACKSERIAL::resizeBuffers_()
{
	{	// clear buffer (completely)
		std::vector<std::vector<Float> > tmp;
		tmp.swap( buffer_ );
	}

	if (layer_.size()<2) return;

	buffer_.resize(layer_.size()-1);

	size_t i = 0;
	for (auto b = buffer_.begin(); b != buffer_.end(); ++b, ++i)
	{
		// resize layer if no fit with previous layer
		if (i>0 && layer_[i]->nrIn() != layer_[i-1]->nrOut())
			layer_[i]->resize(layer_[i-1]->nrOut(), layer_[i]->nrOut());

		// alloc imidiate buffer
		b->resize(layer_[i]->nrOut());
	}

	for (auto b = buffer_.begin(); b != buffer_.end(); ++b)
		std::cout << "buffer: " << b->size() << "\n";
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_STACKSERIAL::fprop(Float * input, Float * output)
{
	if (layer_.empty()) return;

	// fprop single layer
	if (layer_.size()==1)
	{
		layer_[0]->fprop(input, output);
		return;
	}

	// fprob first layer
	layer_[0]->fprop(input, &buffer_[0][0]);

	// fprob hidden layers
	for (size_t i = 1; i<layer_.size()-1; ++i)
	{
		layer_[i]->fprop(&buffer_[i-1][0], &buffer_[i][0]);
	}

	// fprob last layer
	layer_[layer_.size()-1]->fprop(&buffer_[buffer_.size()-1][0], output);
}


MNN_TEMPLATE
void MNN_STACKSERIAL::bprop(Float * error, Float * error_output, Float global_learn_rate)
{
	if (layer_.empty()) return;

	// bprop single layer
	if (layer_.size()==1)
	{
		layer_[0]->bprop(error, error_output, global_learn_rate);
		return;
	}

	// bprob last layer
	layer_[layer_.size()-1]->bprop(error, &buffer_[buffer_.size()-1][0], global_learn_rate);

	// bprob hidden layers
	for (size_t i = layer_.size()-2; i > 0; --i)
	{
//		std::cout << "bprob layer["<<i<<"] with buffer["<<i<<"] > buffer["<<(i-1)<<"]\n";
		layer_[i]->bprop(&buffer_[i][0], &buffer_[i-1][0], global_learn_rate);
	}

	// bprob first layer
	layer_[0]->bprop(&buffer_[0][0], error_output, global_learn_rate);
}


// ----------- info -----------------------

MNN_TEMPLATE
void MNN_STACKSERIAL::info(std::ostream &out) const
{
	out << "name : " << name() << "\n";
	size_t k = 1;
	for (auto l = layer_.begin(); l != layer_.end(); ++l, ++k)
	{
		out << "-- layer " << k << " --\n";
		(*l)->info(out);
	}
}

MNN_TEMPLATE
void MNN_STACKSERIAL::dump(std::ostream &out) const
{
	size_t k = 0;
	for (auto l = layer_.begin(); l != layer_.end(); ++l, ++k)
	{
		out << "-- layer " << k << " --\n";
		(*l)->dump(out);
	}
}






#undef MNN_TEMPLATE
#undef MNN_STACKSERIAL

