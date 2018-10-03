function [a,da]=vekt(x,y)
    n = length(x);
    D = sum(x.^2) - (1/n)*(sum(x))^2;
    E = sum(x.*y) - (1/n)*(sum(x))*(sum(y));
    a = (E/D)
    b = mean(y) - a*mean(x);
    f = a*x + b;
    di = y - a*x - b;
    da = sqrt((1/(n-2))*(sum(di.^2)/D))
    plot(x,f,'o')