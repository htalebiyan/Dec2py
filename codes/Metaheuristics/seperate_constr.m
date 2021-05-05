function [A, b, Aeq, beq]=seperate_constr(constr_rhs, constr_sense, Afull)
A = [];
b = [];
Aeq = [];
beq = [];
constrNames = {fieldnames(constr_rhs)};
constrNames = constrNames{1,1};
constrRhs = cell2mat(struct2cell(constr_rhs));
constrSense = cell2mat(struct2cell(constr_sense));
numConstr = numel(constrNames);
for i=1:numConstr 
    sense = constrSense(i);
    if sense == '='
        beq = [beq; constrRhs(i)];
        Aeq = [Aeq; Afull(i,:)];
    elseif sense == '<'
        b = [b; constrRhs(i)];
        A = [A; Afull(i,:)];
    elseif sense == '>'
        b = [b; -constrRhs(i)];
        A = [A; -Afull(i,:)];
    else
        sprintf('Wrong sense for constraints')
    end
end
