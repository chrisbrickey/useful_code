#assumes we have defined a Node class where nodes have a @value that is accessible
#...and have added children to nodes to make a tree
#.children returns array of the children nodes
#.parent return the node of the parent (or nil if no parent)
#We start at the root node and work our way down

#the 'root' will represent any node in the tree because we're using recursion, starts as actual root, changes with each recursion
def dfs_object(root_node, target)
  #two base cases
  return root_node if root_node.value == target
  # return nil if root_node.parent.nil? #when there are no parents, we know we're back at the actual root of the tree

  root_node.children.each do |child_node|
    result = dfs(child_node, target)

    #returning nil at this point would cut short
    if result #is not nil
      return result
    end
  end

  nil
end


def bfs_object(root_node, target)
  local_arr = [root_node]
  until local_arr.empty?
    potential_target = local_arr.shift
    return potential_target if potential_target.value == target
    local_arr += potential_target.children
  end
  nil
end
