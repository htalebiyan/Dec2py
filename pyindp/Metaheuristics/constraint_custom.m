function [c, ceq] = constraint_custom(x,constr_index,A)
b = cell2mat(struct2cell(constr_index));
c = A*x-b;
ceq = [];