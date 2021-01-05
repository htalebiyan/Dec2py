function y = obj_function(x, obj_coeffs)
coeffs = cell2mat(obj_coeffs);
y = coeffs * x';