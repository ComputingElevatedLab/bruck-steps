#include <vector>

int myPow(int x, unsigned int p)
{
    if (p == 0)
    {
        return 1;
    }
    else if (p == 1)
    {
        return x;
    }

    int tmp = myPow(x, p / 2);
    if (p % 2 == 0)
    {
        return tmp * tmp;
    }
    else
    {
        return x * tmp * tmp;
    }
}

std::vector<int> convert10tob(int w, int N, int b)
{
    std::vector<int> v(w);
    int i = 0;
    while (N)
    {
        v[i++] = (N % b);
        N /= b;
    }
    return v;
}