function label=spamOrHam(data, dtree)
    if dtree.attr == 0 
        label = dtree.label;
        return;
    end
    if data(dtree.attr) < dtree.splitpoint
        label = spamOrHam(data, dtree.left);
    else
        label = spamOrHam(data, dtree.right);
    end