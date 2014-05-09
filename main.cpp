#include <iostream>
#include <algorithm>
#include <mnn/mnn.h>


template <typename Float, class Net>
void simple_test(Net& net, size_t nrIn, size_t nrSamples, std::vector<Float>& input, std::vector<Float>& result)
{
	Float output;

	net.resize(nrIn, 1);
	net.brainwash();
	net.info();
	net.dump();

	// stochastic gradient descent
	std::cout << "training simple pattern with "<<nrIn<<" inputs...\n";
	Float error = 1.0, e;
	size_t count = 0;

	while (error > 0.01 && count<100000)
	{
		// choose pattern
		size_t nr = rand()%nrSamples;
		// feed forward
		net.fprop(&input[nr*nrIn], &output);
		// compare
		e = result[nr] - output;
		// average error over time
		error += 0.01 * (fabs(e) - error);
		// train
		net.bprop(&e);
		std::cout << "av. error " << error << " out " << output << "           \r";
		//net.print();
		count++;
	}
	std::cout << "took "<<count<<" epochs                                      \n";

	std::cout << "testing...\n";
	error = 0.0;
	for (size_t i=0;i<nrSamples;i++)
	{
		net.fprop(&input[i*nrIn], &output);
		// compare
		e = fabs(result[i] - output);
		error += e;
		for (size_t j=0;j<nrIn;j++)
			std::cout << input[i*nrIn+j] << ", ";
		std::cout << " =\t" << output << " (abs. error " << e << ")\n";
	}
	error /= nrSamples;
	std::cout << "average abs. error " << error << "\n";

}

template <typename Float, class Net>
void test_some_pattern(Net& net)
{
	const size_t nrIn = 3;
	const size_t nrSamples = 5;
	std::vector<Float> input(
	{ 	0.0, 0.0, 0.0,
		0.2, 0.3, 0.2,
		0.7, 0.6, 0.5,
		0.9, 0.8, 0.9,
		0.1, 0.5, 0.9
	} );
	std::vector<Float> result(
	{
		0.0,
		0.23,
		0.6,
		0.86,
		0.5
	});
	simple_test(net, nrIn, nrSamples, input, result);
};

template <typename Float, class Net>
void test_xor_pattern(Net& net)
{
	const size_t nrIn = 3;
	const size_t nrSamples = 4;
	std::vector<Float> input(
	{ 	1, 0.1, 0.1,
		1, 0.1, 0.9,
		1, 0.9, 0.1,
		1, 0.9, 0.9
	} );
	std::vector<Float> result(
	{
		0.1,
		0.9,
		0.9,
		0.1
	});
	simple_test(net, nrIn, nrSamples, input, result);
};







template <typename Float, class Net>
void xor_test(Net& net)
{

	const size_t nrIn = 2;
	const size_t nrSamples = 4;
	const Float X = 0.9;
	const Float O = 0.1;
	std::vector<Float> xor_input(
	{
		O, O,
		O, X,
		X, O,
		X, X
	});
	std::vector<Float> xor_result(nrSamples);

	for (size_t i=0;i<nrSamples;i++)
	{
		size_t k = (xor_input[i*nrIn]>0.5);
		for (size_t j=1;j<nrIn;j++)
			k = k xor (xor_input[i*nrIn+j]>0.5);
		xor_result[i] = (k)? X : O;
	}

	Float output;

	net.resize(nrIn, 1);
	net.brainwash();
	net.info();
	net.dump();

	// stochastic gradient descent
	std::cout << "training "<<nrIn<<"-XOR pattern...\n";
	Float error = 1.0, e;
	size_t count = 0;

	while (error > 0.01 && count<100000)
	{
		// choose pattern
		size_t nr = rand()%nrSamples;
		// feed forward
		net.fprop(&xor_input[nr*nrIn], &output);
		// compare
		e = xor_result[nr] - output;
		// average error over time
		error += 0.01 * (fabs(e) - error);
		// train
		net.bprop(&e);
		std::cout << "av. error " << error << " out " << output << "           \r";
		//net.print();
		count++;
	}
	std::cout << "took "<<count<<" epochs                                      \n";

	std::cout << "testing...\n";
	error = 0.0;
	for (size_t i=0;i<nrSamples;i++)
	{
		net.fprop(&xor_input[i*nrIn], &output);
		// compare
		e = fabs(xor_result[i] - output);
		error += e;
		for (size_t j=0;j<nrIn;j++)
			std::cout << xor_input[i*nrIn+j] << ", ";
		std::cout << " =\t" << output << " (abs. error " << e << ")\n";
	}
	error /= nrSamples;
	std::cout << "average abs. error " << error << "\n";
}









int main()
{
	srand(time(NULL));

	//MNN::Perceptron<float, MNN::Activation::Linear> net;

	MNN::StackSerial<float> net;
	net.add( new MNN::Perceptron<float, MNN::Activation::Tanh>(10,10, 1) );
	net.add( new MNN::Perceptron<float, MNN::Activation::Tanh>(10,10, 1) );
	net.add( new MNN::Perceptron<float, MNN::Activation::Linear>(10,10, 0.1) );


	test_xor_pattern<float>(net);

	return 0;
}
