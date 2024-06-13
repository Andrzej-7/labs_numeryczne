//lab 1


#include <iostream>

int main() {
    float a = 0.1;
    float b = 0.2;
    float c = 0.3;

    float result1 = (a + b) + c;
    float result2 = a + (b + c);

    std::cout << "(a + b) + c = " << result1 << std::endl;
    std::cout << "a + (b + c) = " << result2 << std::endl;

    if (result1 == result2) {
        std::cout << "Kolejność dodawania jest przemienna." << std::endl;
    } else {
        std::cout << "Kolejność dodawania nie jest przemienna." << std::endl;
    }

    return 0;
}

//lab2

#include <iostream>
#include <cmath>

float distanceEuclidean(float x1, float y1, float x2, float y2) {
    return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

float distanceManhattan(float x1, float y1, float x2, float y2) {
    return abs(x2 - x1) + abs(y2 - y1);
}

float distanceChebyshev(float x1, float y1, float x2, float y2) {
    return std::max(abs(x2 - x1), abs(y2 - y1));
}

float distanceMinkowski(float x1, float y1, float x2, float y2) {
    return 2 * abs(x2 - x1) + abs(y2 - y1);
}

int main() {
    float x1, y1, x2, y2;
    std::cout << "Podaj wspolrzedne pierwszego punktu (x y): ";
    std::cin >> x1 >> y1;
    std::cout << "Podaj wspolrzedne drugiego punktu (x y): ";
    std::cin >> x2 >> y2;

    float distanceEuc = distanceEuclidean(x1, y1, x2, y2);
    float distanceMan = distanceManhattan(x1, y1, x2, y2);
    float distanceCheb = distanceChebyshev(x1, y1, x2, y2);
    float distanceMink = distanceMinkowski(x1, y1, x2, y2);

    std::cout << "Odleglosc euklidesowa: " << distanceEuc << std::endl;
    std::cout << "Odleglosc Manhattan: " << distanceMan << std::endl;
    std::cout << "Odleglosc rzezi: " << distanceCheb << std::endl;
    std::cout << "Odleglosc kolejowa: " << distanceMink << std::endl;

    return 0;
}



//mnozenie macierzy
#include <iostream>
#include <vector>

using namespace std;

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw invalid_argument("Number of columns in A must be equal to number of rows in B");
    }

    vector<vector<int>> C(rowsA, vector<int>(colsB, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}

int main() {
    vector<vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };

    vector<vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    try {
        vector<vector<int>> C = multiplyMatrices(A, B);
        cout << "Matrix A * B:" << endl;
        printMatrix(C);

        if (A[0].size() == B.size()) {
            vector<vector<int>> D = multiplyMatrices(B, A);
            cout << "Matrix B * A:" << endl;
            printMatrix(D);
            cout << "Multiplication is commutative: " << (C == D) << endl;
        }

        vector<vector<int>> C1 = {
            {13, 14},
            {15, 16}
        };

        if (B[0].size() == C1.size()) {
            vector<vector<int>> temp = multiplyMatrices(B, C1);
            vector<vector<int>> leftAssoc = multiplyMatrices(A, temp);

            vector<vector<int>> temp2 = multiplyMatrices(A, B);
            vector<vector<int>> rightAssoc = multiplyMatrices(temp2, C1);

            cout << "Associativity check:" << endl;
            cout << "(A * (B * C1)):" << endl;
            printMatrix(leftAssoc);
            cout << "((A * B) * C1):" << endl;
            printMatrix(rightAssoc);
            cout << "Multiplication is associative: " << (leftAssoc == rightAssoc) << endl;
        }

    } catch (const invalid_argument& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}


//lab 6 metoda busekcji

#include <iostream>
#include <cmath>
#include <functional>

double bisectionMethod(std::function<double(double)> func, double a, double b, double tol = 1e-6, int maxIter = 1000) {
    if (func(a) * func(b) >= 0) {
        std::cerr << "Invalid interval: f(a) and f(b) must have opposite signs." << std::endl;
        return NAN;
    }

    double mid = a;
    for (int i = 0; i < maxIter; ++i) {
        mid = (a + b) / 2.0;
        double f_mid = func(mid);

        if (fabs(f_mid) < tol) {
            return mid;
        } else if (func(a) * f_mid < 0) {
            b = mid;
        } else {
            a = mid;
        }
    }

    std::cerr << "Maximum iterations reached without finding the root." << std::endl;
    return mid;
}

int main() {
    auto f1 = [](double x) { return x * x - 4; };
    auto f2 = [](double x) { return sin(x) - 0.5; };

    double a1 = 0.0, b1 = 2.2;
    double a2 = 0.0, b2 = 2.2;

    double tol = 1e-6;
    int maxIter = 1000;

    double root1 = bisectionMethod(f1, a1, b1, tol, maxIter);
    double root2 = bisectionMethod(f2, a2, b2, tol, maxIter);

    std::cout << "Root of f(x) = x^2 - 4 in [0, 2.2]: " << root1 << std::endl;
    std::cout << "Root of f(x) = sin(x) - 0.5 in [0, 2.2]: " << root2 << std::endl;

    return 0;
}

//lab7


#include <iostream>
#include <vector>

using namespace std;

// Function to perform Lagrange interpolation
double lagrangeInterpolation(const vector<double>& x, const vector<double>& y, double xi) {
    long n = x.size();
    double result = 0.0;

    for (int i = 0; i < n; ++i) {
        double term = y[i];
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                term *= (xi - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }

    return result;
}

int main() {
    vector<double> x = {-1.4, -1.0, 0.0, 1.0, 2.0, 2.2, 2.5, 2.7, 3.0, 3.2};
    vector<double> y = {11.95, 1.85, 1.0, 0.54, 0.17, 0.31, 0.57, 0.76, 0.97, 0.99};

    double xi = 1.5;

    double yi = lagrangeInterpolation(x, y, xi);

    cout << "Interpolated value at x = " << xi << " is y = " << yi << endl;

    return 0;
}



//lab 8


#include <iostream>
#include <cmath>
#include <chrono>


unsigned long long factorial(int n) {
    if (n == 0 || n == 1) return 1;
    unsigned long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double maclaurinExponential(double x, int terms) {
    double sum = 1.0; 
    for (int n = 1; n < terms; ++n) {
        sum += pow(x, n) / factorial(n);
    }
    return sum;
}

int main() {
    double x = 1.0; 
    int terms = 20; 

    auto start_maclaurin = std::chrono::high_resolution_clock::now();
    double maclaurin_result = maclaurinExponential(x, terms);
    auto end_maclaurin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_maclaurin = end_maclaurin - start_maclaurin;

    auto start_library = std::chrono::high_resolution_clock::now();
    double library_result = std::exp(x);
    auto end_library = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_library = end_library - start_library;

    std::cout << "Maclaurin series result for e^" << x << " = " << maclaurin_result << std::endl;
    std::cout << "Library function result for e^" << x << " = " << library_result << std::endl;
    std::cout << "Time taken by Maclaurin series: " << elapsed_maclaurin.count() << " seconds" << std::endl;
    std::cout << "Time taken by library function: " << elapsed_library.count() << " seconds" << std::endl;

    return 0;
}


