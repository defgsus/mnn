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
StackSerial<Float>& MNN_STACKSERIAL::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const StackSerial<Float>*>(&layer);
    if (!net)
        return *this;

    Stack<Float>::clearLayers();

    for (size_t i = 0; i < net->numLayers(); ++i)
    {
        auto l = net->layer(i)->getCopy();
        Stack<Float>::addLayer(l);
    }

    return *this;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_STACKSERIAL::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // dimension
    s << " " << Stack<Float>::numLayers() << " ";
    // each layer
    for (auto l : Stack<Float>::layer_)
    {
        s << "\n";
        l->serialize(s);
    }
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
    if (numLay != Stack<Float>::numLayers())
        MNN_EXCEPTION("Number of layers does not match in " << name()
                      << ", expected " << Stack<Float>::numLayers()
                      << ", found " << numLay);
    // each layer
    for (auto l : Stack<Float>::layer_)
        l->deserialize(s);

    updateLayers();
}



// ----------- Stack interface -----------


MNN_TEMPLATE
void MNN_STACKSERIAL::updateLayers()
{
	{	// clear buffer (completely)
        std::vector<std::vector<Float>> tmp;
		tmp.swap( buffer_ );
	}

    if (Stack<Float>::layer_.size()<2)
        return;

    buffer_.resize(Stack<Float>::layer_.size()-1);

    for (size_t i = 0; i < Stack<Float>::layer_.size(); ++i)
	{
		// resize layer if no fit with previous layer
        if (i > 0 && Stack<Float>::layer_[i]->numIn() != Stack<Float>::layer_[i-1]->numOut())
            Stack<Float>::layer_[i]->resize(
                        Stack<Float>::layer_[i-1]->numOut(),
                        Stack<Float>::layer_[i]->numOut());

        // alloc intermediate buffer
        if (i < buffer_.size())
            buffer_[i]. resize(Stack<Float>::layer_[i]->numOut());
	}

//	for (auto b = buffer_.begin(); b != buffer_.end(); ++b)
//		std::cout << "buffer: " << b->size() << "\n";
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_STACKSERIAL::fprop(const Float * input, Float * output)
{
    if (Stack<Float>::layer_.empty()) return;

	// fprop single layer
    if (Stack<Float>::layer_.size()==1)
	{
        Stack<Float>::layer_[0]->fprop(input, output);
		return;
	}

	// fprob first layer
    Stack<Float>::layer_[0]->fprop(input, &buffer_[0][0]);

	// fprob hidden layers
    for (size_t i = 1; i<Stack<Float>::layer_.size()-1; ++i)
	{
        Stack<Float>::layer_[i]->fprop(&buffer_[i-1][0], &buffer_[i][0]);
	}

	// fprob last layer
    Stack<Float>::layer_[Stack<Float>::layer_.size()-1]->fprop(
                &buffer_[buffer_.size()-1][0], output);
}


MNN_TEMPLATE
void MNN_STACKSERIAL::bprop(const Float * error, Float * error_output,
                            Float global_learn_rate)
{
    if (Stack<Float>::layer_.empty())
        return;

	// bprop single layer
    if (Stack<Float>::layer_.size()==1)
	{
        Stack<Float>::layer_[0]->bprop(error, error_output, global_learn_rate);
		return;
	}

	// bprob last layer
    Stack<Float>::layer_[Stack<Float>::layer_.size()-1]->bprop(
                error, &buffer_[buffer_.size()-1][0], global_learn_rate);

	// bprob hidden layers
    for (size_t i = Stack<Float>::layer_.size()-2; i > 0; --i)
	{
//		std::cout << "bprob layer["<<i<<"] with buffer["<<i<<"] > buffer["<<(i-1)<<"]\n";
        Stack<Float>::layer_[i]->bprop(&buffer_[i][0], &buffer_[i-1][0], global_learn_rate);
	}

	// bprob first layer
    Stack<Float>::layer_[0]->bprop(&buffer_[0][0], error_output, global_learn_rate);
}


// ----------- info -----------------------


MNN_TEMPLATE
void MNN_STACKSERIAL::info(std::ostream &out, const std::string& pf) const
{
    out <<         pf << "name      : " << name()
        << "\n" << pf << "layout    : ";
    if (Stack<Float>::layer_.empty())
        out << "empty";
    else
    {
        out << Stack<Float>::layer_[0]->numIn();
        for (auto l : Stack<Float>::layer_)
            out << " - " << l->numOut();
    }
    out << "\n" << pf << "parameters: " << Stack<Float>::numParameters()
        << "\n";
	size_t k = 1;
    for (auto l = Stack<Float>::layer_.begin(); l != Stack<Float>::layer_.end(); ++l, ++k)
	{
        out << pf << "-- layer " << k << " --\n";
        (*l)->info(out, pf + "  ");
	}
}

MNN_TEMPLATE
void MNN_STACKSERIAL::dump(std::ostream &out) const
{
	size_t k = 0;
    for (auto l = Stack<Float>::layer_.begin(); l != Stack<Float>::layer_.end(); ++l, ++k)
	{
		out << "-- layer " << k << " --\n";
		(*l)->dump(out);
	}
}






#undef MNN_TEMPLATE
#undef MNN_STACKSERIAL

