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
    for (auto l : layer_)
        delete l;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_STACKSERIAL::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // dimension
    s << " " << numLayer();
    // each layer
    for (auto l : layer_)
        l->serialize(s);
}

MNN_TEMPLATE
void MNN_STACKSERIAL::deserialize(std::istream& s)
{
    std::string str;
    s >> str;
    if (str != id())
        MNN_EXCEPTION("Expected '" << id()
                      << "' in stream, found '" << str << "'");
    // version
    int ver;
    s >> ver;
    if (ver > 1)
        MNN_EXCEPTION("Wrong version in " << name());

    // dimension
    size_t numLay;
    s >> numLay;
    if (numLay != numLayer())
        MNN_EXCEPTION("Number of layers does not match in " << name()
                      << ", expected " << numLayer() << ", found " << numLay);
    // each layer
    for (auto l : layer_)
        l->deserialize(s);

    resizeBuffers_();
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_STACKSERIAL::resize(size_t numIn, size_t numOut)
{
	if (layer_.empty()) return;

    layer_[0]->resize(numIn, layer_[0]->numOut());
    layer_[layer_.size()-1]->resize(layer_[layer_.size()-1]->numIn(), numOut);

	resizeBuffers_();
}

MNN_TEMPLATE
void MNN_STACKSERIAL::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    if (layer_.empty())
        return;

    if (nrIn > layer_.front()->numIn())
        layer_.front()->grow(nrIn, layer_.front()->numOut(), randomDev);
    if (nrOut > layer_.back()->numOut())
        layer_.back()->grow(layer_.back()->numIn(), nrOut, randomDev);

    resizeBuffers_();
}

MNN_TEMPLATE
size_t MNN_STACKSERIAL::numIn() const
{
    return (layer_.empty())? 0 : layer_[0]->numIn();
}

MNN_TEMPLATE
size_t MNN_STACKSERIAL::numOut() const
{
    return (layer_.empty())? 0 : layer_.back()->numOut();
}


MNN_TEMPLATE
void MNN_STACKSERIAL::brainwash(Float amp)
{
    for (auto l : layer_)
        l->brainwash(amp);
}

// ----------- layer interface -----------

MNN_TEMPLATE
size_t MNN_STACKSERIAL::numLayer() const
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
void MNN_STACKSERIAL::insert(size_t index, Layer<Float> * layer)
{
    if (index >= layer_.size())
        layer_.push_back(layer);
    else
        layer_.insert(layer_.begin() + index, layer);
    resizeBuffers_();
}

MNN_TEMPLATE
void MNN_STACKSERIAL::resizeBuffers_()
{
	{	// clear buffer (completely)
        std::vector<std::vector<Float>> tmp;
		tmp.swap( buffer_ );
	}

    if (layer_.size()<2)
        return;

	buffer_.resize(layer_.size()-1);

    for (size_t i = 0; i < layer_.size(); ++i)
	{
		// resize layer if no fit with previous layer
        if (i > 0 && layer_[i]->numIn() != layer_[i-1]->numOut())
            layer_[i]->resize(layer_[i-1]->numOut(), layer_[i]->numOut());

        // alloc immediate buffer
        if (i < buffer_.size())
            buffer_[i]. resize(layer_[i]->numOut());
	}

//	for (auto b = buffer_.begin(); b != buffer_.end(); ++b)
//		std::cout << "buffer: " << b->size() << "\n";
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_STACKSERIAL::fprop(const Float * input, Float * output)
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
void MNN_STACKSERIAL::bprop(const Float * error, Float * error_output,
                            Float global_learn_rate)
{
    if (layer_.empty())
        return;

	// bprop single layer
	if (layer_.size()==1)
	{
		layer_[0]->bprop(error, error_output, global_learn_rate);
		return;
	}

	// bprob last layer
    layer_[layer_.size()-1]->bprop(
                error, &buffer_[buffer_.size()-1][0], global_learn_rate);

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
Float MNN_STACKSERIAL::getWeightAverage() const
{
    Float a = 0.;
    for (auto l : layer_)
        a += l->getWeightAverage();
    if (!layer_.empty())
        a /= layer_.size();
    return a;
}

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

