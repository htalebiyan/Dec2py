function [x,fval]=main(var_index, constr_rhs, constr_sense, obj_coeffs)
rng default % For reproducibility
Asparse = load('arrdata.mat');
Afull = full(Asparse.A);
ObjectiveFunction = @(x)obj_function(x, obj_coeffs);
nvars = size(obj_coeffs,2);    % Number of variables
[UB,LB,IntCon] = bounds(var_index);
[A, b, Aeq, beq] = seperate_constr(constr_rhs, constr_sense, Afull);
% ConstraintFunction = @(x)constraint_custom(x,constr_rhs, constr_sense, Afull);

%% run the GA solver.
options = optimoptions('ga','PlotFcn', @gaplotbestf);
[x,fval,exitflag,output] = ga(ObjectiveFunction,nvars,A,b,Aeq,beq,LB,UB, ...
                                [],IntCon,options)
disp(x);
disp(fval);
disp(exitflag);
disp(output);
disp(A*x' - b);
disp(Aeq*x' - beq);