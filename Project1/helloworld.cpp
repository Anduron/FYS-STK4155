#include <iostream>
#include <cmath>
using namespace std;

int main()
{
	int r;
	double s;
	for (int i = 1; i < 10; i = i + 2)
	{
		r = i;
		s = sin(r);
		cout << "r = " << r << ", " << "s = " << s << "\n";

	}
	cout << "Hello World" << endl;
	return 0;
}
