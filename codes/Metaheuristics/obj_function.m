function y = obj_function(x, obj_coeffs, Ai, bi, Ae, be, lambda)
coeffs = cell2mat(obj_coeffs);
y = coeffs*x'+ sum(abs(Ae*x'-be))*lambda;
% y = y + sum(Ai*x'-bi);%*lambda;