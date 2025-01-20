import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import norm
from autograd.misc.optimizers import sgd, adam
from ssm.optimizers import adamc, lbfgs, bfgs, rmsprop
from autograd import grad
from autograd.numpy.numpy_boxes import ArrayBox, SequenceBox

from ssm.util import ensure_args_are_lists

from itertools import combinations, product

def lists_to_tuples(obj):
    if isinstance(obj, list):
        # If the object is a list, recursively convert its elements
        return tuple(lists_to_tuples(item) for item in obj)
    else:
        # If it's not a list, return it as-is
        return obj

def tuples_to_lists(obj):
    if isinstance(obj, tuple):
        # If the object is a tuple, recursively convert its elements
        return [tuples_to_lists(item) for item in obj]
    else:
        # If it's not a tuple, return it as-is
        return obj
    
def is_valid_slice_for_shape(index_tuple, array_shape):
    """
    Validates whether a given tuple of indices is a valid slice for an array with the given shape.
    """
    num_explicit_dims = sum(1 for idx in index_tuple if idx is not Ellipsis)
    if num_explicit_dims > len(array_shape):
        return False
    
    if ":" in index_tuple:
        colon_pos = index_tuple.index(":")
        #cast as slice(None)
        index_tuple[colon_pos] = slice(None)

    # Handle Ellipsis: fill in missing dimensions
    if Ellipsis in index_tuple:
        ellipsis_pos = index_tuple.index(Ellipsis)
        num_missing_dims = len(array_shape) - num_explicit_dims
        index_tuple = (
            index_tuple[:ellipsis_pos]
            + (slice(None),) * num_missing_dims
            + index_tuple[ellipsis_pos + 1:]
        )

    if len(index_tuple) != len(array_shape):
        return False  # Mismatch in the number of dimensions

    for i, idx in enumerate(index_tuple):
        if isinstance(idx, int):
            if idx < -array_shape[i] or idx >= array_shape[i]:
                return False
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(array_shape[i])
            if start < -array_shape[i] or stop > array_shape[i]:
                return False
        else:
            return False

    return True

def expand_indices(tup, array_shape):
    """
    Expands a tuple containing slices, integers, or Ellipsis into all multi-dimensional indices.
    """
    if not is_valid_slice_for_shape(tup, array_shape):
        raise ValueError(f"Invalid slice {tup} for array shape {array_shape}")
    
    expanded = []
    num_axes = len(array_shape)
    num_explicit = sum(1 for item in tup if item is not Ellipsis)
    
    # Handle colons specifying entire axis
    if ":" in tup:
        colon_pos = tup.index(":")
        #cast as slice(None)
        tup[colon_pos] = slice(None)
    
    # Handle Ellipsis
    if Ellipsis in tup:
        ellipsis_pos = tup.index(Ellipsis)
        num_missing = num_axes - num_explicit
        if num_missing < 0:
            raise ValueError(f"Too many explicit dimensions in tuple {tup} for array shape {array_shape}")
        tup = tup[:ellipsis_pos] + (slice(None),) * num_missing + tup[ellipsis_pos + 1:]

    # Expand dimensions
    for i, item in enumerate(tup):
        if isinstance(item, slice):
            expanded.append(range(item.start or 0, item.stop or array_shape[i], item.step or 1))
        elif isinstance(item, int):
            expanded.append([item])
        else:
            raise ValueError(f"Invalid item {item} in tuple {tup}. Only integers, slices, and Ellipsis are allowed.")

    return set(product(*expanded))

def check_index_overlap_multidimensional_with_shapes(data, shape_dict):
    """
    Checks for overlapping multi-dimensional indices across second-level dictionary keys with overlapping keys.
    Each second-level key can have a different array shape.
    
    Parameters:
        - data: The two-level dictionary to check.
        - shape_dict: A dictionary mapping second-level keys to their respective array shapes.
                      Example: {'param1': (10, 10), 'param2': (5, 5, 5)}
                      
    Returns:
        - A tuple (bool, str) indicating whether overlaps were found and an explanation.
    """
    
    # Iterate over all first-level keys in the data
    top_keys = list(data.keys())

    for key1, key2 in combinations(top_keys, 2):
        dict1 = data[key1]
        dict2 = data[key2]

        # Find common second-level keys
        common_keys = set(dict1.keys()) & set(dict2.keys())

        for common_key in common_keys:
            # Check if the shape for this key exists in shape_dict
            if common_key not in shape_dict:
                raise KeyError(f"Shape for key '{common_key}' not provided in shape_dict.")

            # Get the shape for this common key
            array_shape = shape_dict[common_key]

            # Validate indices for dict1 and dict2
            indices1 = set()
            for tup in (dict1[common_key] if isinstance(dict1[common_key], list) else [dict1[common_key]]):
                indices1.update(expand_indices(tup, array_shape))

            indices2 = set()
            for tup in (dict2[common_key] if isinstance(dict2[common_key], list) else [dict2[common_key]]):
                indices2.update(expand_indices(tup, array_shape))

            # Check for overlap
            overlap = indices1 & indices2
            if overlap:
                return False, f"Overlap found in common key '{common_key}' between {key1} and {key2}: {overlap}"

    return True, "No overlaps found."


def check_overlap_between_tuples_or_lists(tuples1, tuples2, array_shape):
    """
    Checks for overlap between two tuples or lists of tuples of indices.
    
    Parameters:
        - tuples1: A single tuple or a list of tuples representing indices/slices.
        - tuples2: A single tuple or a list of tuples representing indices/slices.
        - array_shape: Shape of the array the indices refer to.
        
    Returns:
        - A tuple (bool, set) indicating whether overlaps were found and the overlapping indices.
    """
    if not isinstance(tuples1, list):
        tuples1 = [tuples1]
    if not isinstance(tuples2, list):
        tuples2 = [tuples2]

    indices1 = set()
    for tup in tuples1:
        indices1.update(expand_indices(tup, array_shape))

    indices2 = set()
    for tup in tuples2:
        indices2.update(expand_indices(tup, array_shape))

    # Check for overlap
    overlap = indices1 & indices2
    return bool(overlap), overlap

def apply_masks(masks, values):
    """
    Function to apply masks to a tuple of arrays and return a tuple of arrays constrained on the masks.
    
    Parameters:
        - masks (tuple): A tuple of boolean arrays.
        - values (tuple): A tuple of array values.
    
    Returns:
        - tuple: A tuple of arrays constrained on the masks.
    """
    assert len(masks) == len(values), "Masks and values tuples must be of the same length."
    constrained_values = tuple(value[mask] for mask, value in zip(masks, values))
    return constrained_values

def reverse_apply_masks(masks, output_arrays, values):
    """
    Function to reverse the process of applying masks. Writes values to the boolean positions of the output arrays.
    
    Parameters:
        - masks (tuple): A tuple of boolean arrays.
        - output_arrays (tuple): A tuple of arrays to write to.
        - values (tuple): A tuple of values to write to the output arrays.
    
    Returns:
        - tuple: The updated output arrays.
    """
    assert len(masks) == len(output_arrays) == len(values), "Masks, output arrays, and values tuples must be of the same length."
    for mask, output_array, value in zip(masks, output_arrays, values):
        output_array[mask] = value

    return output_arrays

def create_lambda_labels(index_slices):
    '''
    Function to create a dictionary of unique integer labels
    for each terminal list entry in the index_slices dictionary.
    '''

    hier_leaf_keys, hier_leaf_values = get_dict_leaf_items(index_slices)
    #check if there are any lists of tuples as leaf values
    labels = []
    lv = 0
    for leaf in hier_leaf_values:
        if isinstance(leaf, list):
            for tup in leaf:
                labels.append(lv)
        else:
            labels.append(lv)
            
        lv += 1
            
    return labels

def dict_depth(d):
    '''
    Function to find the maximum depth of a dictionary
    '''
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0

def get_dict_leaf_items(d):
    """
    Function to get all leaf/terminal values of a dictionary
    """
    leaf_values = []
    leaf_keys = []

    def _get_leaves(d, current_key=()):
        if isinstance(d, dict):
            for key, value in d.items():
                _get_leaves(value, current_key + (key,))
        else:
            leaf_values.append(d)
            leaf_keys.append(current_key)

    _get_leaves(d)
    # check for keys that are tuples of length 1,
    # if so then unpack the tuple
    return leaf_keys, leaf_values

def get_nested_value(d, keys):
    """
    Function to extract values from nested dictionaries using a tuple of keys.
    """
    for key in keys:
        d = d[key]
    return d

def get_nested_tuple_value(t, indices):
    """
    Function to extract values from nested tuples using a tuple of indices.
    """
    for index in indices:
        t = t[index]
    return t

def set_nested_list_value(lst, indices, value):
    """
    Function to set a value in a nested list of lists using a tuple of indices.
    """
    # Navigate to the second-to-last level
    for index in indices[:-1]:
        lst = lst[index]
    # Set the value at the final index
    lst[indices[-1]] = value

def convert_nested_list_to_tuple(nested_list):
    """
    Function to convert all lists in a nested list and the nested list itself to tuples.
    """
    if isinstance(nested_list, list):
        return tuple(convert_nested_list_to_tuple(item) for item in nested_list)
    else:
        return nested_list

class _Hierarchical(object):
    """
    Base class for hierarchical models.  Maintains a parent class and a
    bunch of children with their own perturbed parameters.
    """
    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, frozen=(None,), hierarchical_params=(None,),**kwargs):
        # Variance of child params around parent params
        self.lmbda_prior_mu = lmbda
        self.lmbda = lmbda
        self.lmda_memory = []
        assert isinstance(self.lmbda, (float, int, list, tuple, np.ndarray)), "lmbda must be a float, list, tuple, or ndarray"
        if isinstance(self.lmbda, (float, int)):
            self.num_lambdas = 1
        else:
            self.num_lambdas = len(self.lmbda)
            
        # Top-level parameters (parent)
        self.parent = base_class(*args, **kwargs)
        self._sqrt_lmbdas = self.set_lmbdas()
            
        # Make models for each tag
        self.tags = tags

        
        self.hierarchical_params = hierarchical_params
        if hierarchical_params != (None,):
            #hierarhcical params can be a list specifiying the indices of the parent parameter list
            #to draw the child parameter values around. If tags is a list of tuples,
            #then this indicates that multiple tag labels are used which should have
            #different hierarchical parameter sets. To specify these sets, the 
            #hierarchical_params should be a dictionary with first level keys 
            #that correspond to the tag tuple entries and the second level keys
            #should be ints that specify the parent parameter list indices and the leaf
            #values should be tuples that specify the parent parameter indices.
            assert isinstance(hierarchical_params, (list, dict)), "hierarchical_params must be a list or dictionary"
            
            #check if the entries of tags are tuples or lists they are all the same length
            #tags which are tuples indicate levels of similarity and difference between
            #child distributions. although each tuple denotes a different child distribution
            #if the entries of the tuples are the same, then the child distributions can be
            #made to share certain sets of underlying child distribution parameters. This 
            #effectively allows for imposing hierarchical structures on multiple specific 
            #parameters of the parent distribution. Before with only single tag labels 
            #all children parameters were distinct and only 1 factor could be used to impose
            #a hierarchical structure. now arbitrarily many factors can be used to impose 
            #hierarchical structure and the parameter subsets shared across tag/factor 
            #levels can be specified.
            if isinstance(tags[0], (tuple, list)):
                #check that all tags are tuples or lists
                assert all(isinstance(tags[i], (tuple, list)) for i in range(len(tags))) 
                if not all(len(tag) == len(tags[0]) for tag in tags):
                    raise ValueError("All tags must have the same number of entries")
                
                #now check that there are the same number of hierarchical parameters as
                #there are entries in the tags tuples. In this case this dictionary should 
                #have an integer key for each tag tuple entry and the value should be
                #a tuple of integers or a list of tuples of integers if separate parameter
                #indices for the same parameter type are to be controlled by unique lambdas.
                #The second level keys of hierarchical params should refer to unique list 
                #indices of the parent parameter list of arrays and thus are not larger 
                #than the length of the parent parameter array. 
                #The params property of the model is by definition a tuple of arrays, 
                #so if the hierarchy is to be established over specific axes of 
                #the parameter arrays, then the indices of these axes must be specified 
                #in the hierarchical_params dictionary with the second level dictionary keys 
                #specifying the parameter index of the parent parameter list and leaf values being tuple
                #entries or lists of tuple entries specifying the axes of the parameter array 
                #that are to be shared across the child distributions with the same tag tuple entry.
                ## Example dictionary to specify indices and slices:
                # index_slices = {
                #     0: {1: (0, 1, ...)},  # 2nd list entry, [0, 1, :]
                #     1: {0: (slice(1, 3), slice(2, 4)))}  # 1st list entry, [1:3, 2:4]
                # }
                assert isinstance(hierarchical_params, dict), "If tags are tuples, then hierarchical_params must be a dictionary"
                
                #set up a dictionary of masks that look just like the base
                #parameter arrays to explicitly map the shared parameters
                #this will be helpful because we will only optimize one copy
                #of the shared parameters but will need to update ALL the parameters
                #of the model.
                shared_mask_entry = []
                base_params = self.params
                #initialize the shared mask to be all ones
                # for i, mod in enumerate(base_params):
                #     temp_mask = []
                #     for mod_param in mod:
                #         temp_mask.append(np.zeros_like(mod_param))
                #     shared_mask_entry.append(temp_mask)
                
                # #for each tag tuple entry create a dictionary of shared parameter masks
                # #to indicate which parameters of the model are shared across the child
                # #distributions with the same tag tuple entry.
                # self.shared_mask = {k: copy.deepcopy(shared_mask_entry) for k in range(len(tags[0]))}

                if len(np.unique(self.hierarchical_params.keys())) == len(tags[0]):
                    if not all(isinstance(self.hierarchical_params[k], dict) for k in self.hierarchical_params.keys()):
                        raise ValueError("All tag_params values must be dictionaries with keys \
                                        indicating parent parameter list indices and values that are tuples specifying parameter array indices")
                    
                    for k in list(self.hierarchical_params.keys()):
                        #check that all keys in the next level dictionary are integers
                        if not all(isinstance(k_, int) for k_ in self.hierarchical_params[k].keys()):
                            raise ValueError("If tag_params values are dictionaries, the keys must be integers")
                        #check that the maximum keys is less than the number of parent parameter list entries
                        if max(self.hierarchical_params[k].keys()) >= len(self.parent.params):
                            raise ValueError("The maximum key in the tag_params dictionary must be less than \
                                            the number of parent parameter list entries")
                    
                    #check that the specified hierarchical params indices and slices are valid
                    #i.e. that they are valid slices for the parameter arrays and that
                    #there is no overlap in the indices between the different tag tuple entries
                    shape_dict = {k: self.parent.params[k].shape for k in range(len(self.parent.params))}
                    result, message = check_index_overlap_multidimensional_with_shapes(self.hierarchical_params, shape_dict)
                    if not result:
                        raise ValueError(message)
                    
                else:
                    raise ValueError("The number of hierarchical parameters must \
                                        equal the number of entries in the tags tuples")
                #from hierarchical_params, find all indices that are NOT in the hierarchical_params
                self.not_hierarchical_mask = {}
                hier_leaf_keys, hier_leaf_values = get_dict_leaf_items(self.hierarchical_params)
                if self.num_lambdas != 1:
                    assert len(hier_leaf_keys) == len(self.lmbda), "Number of hierarchical parameters must match the number of lambdas, or use a single lambda for all parameters"
                
                for i in range(len(self.parent.params)):
                    #find any hierarchical_params values that refer to the given parameter index
                    #i.e. regardless of tuple entry (first layer keys) find all the leaf values
                    #behind 'i' in the second level keys of the hierarchical_params dictionary
                    leaf_values_i = [lv for lk, lv in zip(hier_leaf_keys, hier_leaf_values) if lk[1] == i]

                    #if the parameter tuple index is NOT in the hierarchical_params
                    #then ALL the indices of the underlying parameter array are NOT
                    #hierarhcical.
                    if len(leaf_values_i) == 0:
                        self.not_hierarchical_mask[i] = np.ones_like(self.parent.params[i]).astype(bool)
                    #if instead the parameter tuple index is in the hierarchical_params
                    #then check if there are indices of the parameter array that are
                    #NOT in the hierarchical_params. If there are, then these indices
                    #are NOT hierarchical.
                    else:
                        #make a mask to collect where the indices
                        #in the hierarchical_params are NOT in the parameter array
                        total_mask = np.ones_like(self.parent.params[i]).astype(bool)
                        for lv in leaf_values_i:
                            #set the mask to zero for the hierarchical parameters
                            total_mask[lv] = 0
                        
                        self.not_hierarchical_mask[i] = total_mask
            
        self.children = dict()
        for tag in tags:
            self.children[tag] = base_class(*args, **kwargs)
            # ch = self.children[tag] = base_class(*args, **kwargs)
            # ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

        #list of parameters that are free to be optimized
        #this must be defined at the resolution of the 
        #parameter getter method output which is a tuple of
        #tuples of parameter arrays
        self.frozen = frozen
        #initialize a mask to translate the frozen parameter
        #slices into a mask for each parameter array. This will
        #facilitate updating/setting the parameters during optimization.
        self.free_mask = []
        base_params = self.params
        #initialize the free mask to be all ones
        for i, mod in enumerate(base_params):
            fmask_entry = []
            if isinstance(mod, (list, tuple)):
                for mod_param in mod:
                    fmask_entry.append(np.ones_like(mod_param).astype(bool))
            elif isinstance(mod, np.ndarray):
                fmask_entry.append(np.ones_like(mod).astype(bool))
            else:
                raise ValueError("Model parameters must be a tuples of arrays or arrays")
            self.free_mask.append(fmask_entry)
                
        if self.frozen != (None,):#then assume all parameters are free
            assert isinstance(self.frozen, (list, tuple, dict)), "frozen must be a list, tuple, or dict"
            if isinstance(f, (list, tuple)):
                assert all(isinstance(f, int) for f in self.frozen), "free must be a list of integers specifying indices in self.params"
                assert max(self.frozen) < len(base_params), "free index out of range of self.params. see getter method for self.params"
                for f in self.frozen:
                    self.free_mask[f] = np.zeros_like(base_params[f]).astype(bool)
                #check that none of the frozen parameters are hierarchical parameters
                #this will only happen if parent paremeters are frozen, i.e.
                #0 is in the frozen list
                if 0 in self.frozen:
                    for tag_key in self.hierarchical_params:
                        if 0 in self.hierarchical_params[tag_key].keys():
                            raise ValueError("Hierarchical parameters cannot be frozen")
            else:#if frozen is a dictionary
                assert dict_depth(self.frozen) <= 2, "frozen dictionary must have depth <= 2, first level hierarchical param tuple, second level specific child model param tuples."
                assert all(isinstance(f, int) for f in self.frozen.keys()), "frozen keys must be integers specifying indices in the base params tuple, see params property"
                assert max(self.frozen.keys()) < len(base_params), "frozen key out of range of the base params tuple, see params property"
                leaf_keys, leaf_values = get_dict_leaf_items(self.frozen)
                #check that all leaf values are lists of tuples, or None
                assert all(isinstance(f, (list, tuple, type(None))) for f in leaf_values), "frozen leaf values must be lists of tuples or None"
                #now check that each tuple index is valid
                for i, leaf in enumerate(leaf_values):
                    if leaf is not None:
                        if isinstance(leaf, list):
                            for tup in leaf:
                                #get the model parameter array
                                param_array = get_nested_tuple_value(base_params, leaf_keys[i])
                                if not is_valid_slice_for_shape(tup, param_array.shape):
                                    raise ValueError("Invalid slice for array")
                                else:
                                    #set the mask to zero for the frozen parameter
                                    set_val = np.ones_like(param_array).astype(bool)
                                    set_val[tup] = 0
                                    set_nested_list_value(self.free_mask, leaf_keys[i], set_val)
                                    
                        else:
                            tup = leaf#leaf is a tuple indexing the parameter array
                            #get the model parameter array
                            param_array = get_nested_tuple_value(base_params, leaf_keys[i])
                            if not is_valid_slice_for_shape(leaf, param_array.shape):
                                raise ValueError("Invalid slice for array")
                            else:
                                #set the mask to zero for the frozen parameter
                                set_val = np.ones_like(param_array).astype(bool)
                                set_val[tup] = 0
                                set_nested_list_value(self.free_mask, leaf_keys[i], set_val)
                    
                    else:#if the leaf value is None, then it means that ALL the parameter indices at i are frozen
                        param_array = get_nested_tuple_value(base_params, leaf_keys[i])
                        #set the mask to zero for the frozen parameter
                        set_val = np.zeros_like(param_array).astype(bool)
                        set_nested_list_value(self.free_mask, leaf_keys[i], set_val)
                        
                #check that none of the frozen parameters are hierarchical parameters
                #this will only happen if parent paremeters are frozen, i.e.
                # #0 is in the first level of the frozen dictionary
                # if 0 in self.frozen.keys():
                #     for i, inds in enumerate(self.frozen[0].items()):
                #         for tag_key in self.hierarchical_params:
                #             #get all terminal values behind key i
                #             leaf_keys, leaf_values = get_dict_leaf_items(inds)
                #             leaf_values = [lv for lk, lv in zip(leaf_keys, leaf_values) if lk == i]
                #             #flatten the list
                #             leaf_values = [item for sublist in leaf_values for item in sublist]
                        
                #         if leaf_values:#if there are any leaf values
                #             #check if there is overlap between the frozen parameter
                #             #indices and the hierarchical parameter indices for 
                #             #this parameter
                #             pshape = self.parent.params[i].shape
                #             result, message = check_overlap_between_tuples_or_lists(inds, leaf_values, pshape)
                #             if result:
                #                 raise ValueError(message)
                        
        #convert the free mask to a tuple of tuples to make it immutable
        self.free_mask = convert_nested_list_to_tuple(self.free_mask)   
        
    def set_lmbdas(self):
        #Iterate over the list and set every entry in each array equal to the value
        #initialize the lmdas list to have the same number of entries as 
        #there are free distribution parameters
        lmbdas = [0]*self.num_lambdas
        for i in range(self.num_lambdas):
            #if lmbda is a vector or list for each parameter distribution
            if isinstance(self.lmbda, (np.ndarray, list, tuple)):
                lmbdas[i] = self.lmbda[i]
            else:#if lmbda is a scalar
                lmbdas[i] = self.lmbda#np.full_like(arr, self.lmbda)
        lmbdas = np.sqrt(np.array(lmbdas))
        
        return lmbdas

    def stack_child_params(self):
        #collect the parameter arrays as stacked arrays for each parameter type in temp
        temp = [[] for _ in range(len(self.parent.params))]
        num_children = len(self.children)
        for i, tag in enumerate(self.tags):
            for j in range(len(self.children[tag].params)):#for each param type in the tuple
                pvec = self.children[tag].params[j]
                #stack the parameter vector onto the lmdas vector
                if isinstance(pvec, ArrayBox):
                    pvec = pvec._value
                temp[j].append(pvec)

        stacked_params = [np.stack(temp[j],axis=-1) for j in range(len(temp))]
        return stacked_params

    def update_lmbdas(self):
        lambda_lower_bound = 1E-3#hiearchical prior lambda params cannot be negative
        lambda_upper_bound = 10.

        temp = self.stack_child_params()
        temp_lmbdas = [0]*self.num_lambdas
        btwn_state_stds = [0]*self.num_lambdas
        
        for j, spvecs in enumerate(temp):#for each parameter vector in the child params set
            #impute the lambda for each parameter as the standard deviation across all child params
            temp_lmbdas[j] = np.std(spvecs, axis=-1)
            #average across each parameter vector and then take the standard deviation
            btwn_state_stds[j] = np.std(np.mean(spvecs, axis=-1))
            # #ensure that the standard deviation is larger than the lower bound..
            if btwn_state_stds[j] < lambda_lower_bound:
                btwn_state_stds[j] = lambda_lower_bound + lambda_lower_bound/2
            elif btwn_state_stds[j] > lambda_upper_bound:#truncate at
                btwn_state_stds[j] = lambda_upper_bound

        
        #clip temp_lmbdas to be larger than 1E-6
        # temp_lmbdas = [np.mean(np.clip(lmbda, 1E-6, np.inf)) for lmbda in temp_lmbdas]
        temp_lmbdas = [np.clip(np.mean(lmbda), lambda_lower_bound, btwn_state_stds[j]) for j, lmbda in enumerate(temp_lmbdas)]

        # weights = np.exp(np.linspace(-1., 0., self.window_size))
        # weights /= weights.sum()
        # if len(self.lmda_memory) == self.window_size:
        #     temp_lmdas_mem = [0]*len(self.parent.params)
        #     for j in range(len(temp_lmdas)):#for each parameter vector in the child params set
        #         #moving average over all the stored lambdas in the window
        #         #collect the jth parameter vector across all child dists
        #         clams = [lmda_[j] for lmda_ in self.lmda_memory]
        #         temp_lmdas_mem[j] = np.average(clams, axis=-1)#, weights=weights)

        #     self.lmda_memory.pop(0)#remove the oldest set of parameter vector stds across child dists
        #     self.lmda_memory.append(temp_lmdas_mem)#add the new set of parameter vector stds across child dists
        #     new_lmdas = temp_lmdas_mem
        # else:
        #     self.lmda_memory.append(temp_lmdas)#save the standard deviations for the next iteration
        #     new_lmdas = temp_lmdas
        self.lmda_memory.append(temp_lmbdas)#save the standard deviations for the next iteration
        new_lmbdas = temp_lmbdas
        
        self._sqrt_lmbdas = np.array(new_lmbdas)
        # return tuple(temp_lmdas)

    @property
    def lambdas(self):
        return self._sqrt_lmbdas**2
    
    
    def extract_shared_params(self):
        '''
        Here we need to extract a single set of parameters for
        each unique tag tuple entry value. So first for each
        tag tuple entry find the unique set of values and then
        for each of these extract the parameter values from one
        of the child distributions that are hierarchically linked
        to the parent distribution parameter values. All other child
        distribution parameters are set equal to the parent distribution
        parameter values so extract the parent parameter values for these
        parameters.
        '''
        #first find the unique tag tuple entry values
        # Transpose the list of tag tuples
        transposed = zip(*self.tags)
        # Get unique values for each index
        unique_values = tuple(list(set(values)) for values in transposed)
        # Now iterate over the tuple indices and add a copy of the 
        # parent parameter values hierarchically linked to that tuple
        # index for each unique value for that tuple index in unique_values
        prms = apply_masks(self.free_mask[0], self.parent.params)#copy.deepcopy(apply_masks(self.free_mask[0], self.parent.params))
        #check for empty arrays. if so, drop these from prms. this can happen
        #if all parameters at a tuple index are frozen.
        prms = [p for p in prms if p.size > 0]
        extracted_params = {'parent': prms}
        # test_labels = ['parent']
        for i, uvals in enumerate(unique_values):
            extracted_params[i] = {uval:[] for uval in uvals}
            for uval in uvals:
                #get all parent parameter indices for this tag tuple entry
                #this will be a dictionary with keys corresponding to the
                #parent parameter indices and values that are tuples of
                #parameter array indices.
                tag_param_inds = self.hierarchical_params[i]
                #for each referenced parent parameter tuple entry
                #grab a child model parameter value. All the
                #child model parameter values for this tag tuple entry
                #and unique value will be the same, so just grab the
                #first one.
                all_child_tags = [tag for tag in self.tags if tag[i] == uval]
                ref_tag = all_child_tags[0]
                for ip, pinds in tag_param_inds.items():
                    extracted_params[i][uval].append(self.children[ref_tag].params[ip][pinds])#copy.deepcopy(self.children[ref_tag].params[ip][pinds])
                    # test_labels.append((i,uval,ref_tag))
        if self.lambda_update == 'optimized':
            extracted_params['lmbdas'] = [self._sqrt_lmbdas]#copy.deepcopy(self._sqrt_lmbdas)
            # test_labels.append('lmbdas')
                
        return extracted_params
    
    def deflate_params(self):
        #extract params from all parent and child distributions
        prms = self.params

        #if hierarchical params are specified, then extract
        #only the one copy of the parameter values for each
        #unique tag tuple entry value. This will handle
        #frozen parameters internally. This is because
        #frozen parameter indices cannot overlap with the
        #hierarchical parameters indices. This means that
        #child parameter values that are not hierarchically
        #linked to the parent distributions are already effectively
        #frozen because they are just set equal to the parent distribution
        #value. So when accounting for frozen parameters only the parent
        #distribution parameter values need to be considered.
        if self.hierarchical_params != (None,):
            prms_dict = self.extract_shared_params()
            #unpack the dictionary into a list of parameter arrays
            self.prms_keys, prms = get_dict_leaf_items(prms_dict)
            #turn prms into tuples of tuples
            prms = lists_to_tuples(prms)
        #if frozen is specified then we must handle freezing the 
        #parent and child distribution parameter values. Only
        #extracting the free parameter values.
        elif self.frozen != (None,):
            prms = tuples_to_lists(prms)
            prms = extract_free_params(prms)
            #turn prms into tuples of tuples
            prms = lists_to_tuples(prms)
        
        return prms
    
    @property    
    def params(self):
        prms = (self.parent.params,)
        for tag in self.tags:
            prms += (self.children[tag].params,)
        if self.lambda_update == 'optimized':
            prms += ((self._sqrt_lmbdas,),)
        return prms

    @params.setter
    def params(self, value):
        assert len(value) == len(self.params), "Number of parameters must match"
        self.parent.params = value[0]
        if self.lambda_update == 'optimized':
            for tag, prms in zip(self.tags, value[1:-1]):
                self.children[tag].params = prms
            self._sqrt_lmbdas = value[-1][0]
        else:
            for tag, prms in zip(self.tags, value[1:]):
                self.children[tag].params = prms
    
    def inflate_params(self, params):
        #get copy of the full parameter set
        # base_params = self.params
        #cast tuples to lists for mutability as we
        #want to be able to set the parameters.
        # base_params = tuples_to_lists(base_params)
        # #create labels for the parent, child and lambda parameters
        base_param_labels = ['parent']
        for tag in self.tags:
            base_param_labels.append(tag)
        base_param_labels.append('lambdas')
        
        # if isinstance(params, SequenceBox):
        #     params = params._value
        
        #iterate over the deflated/reduced params passed
        for i, val in enumerate(params):
            #get the key for where the deflated param
            #refers to in the original parameter array
            pkey = self.prms_keys[i]
            if pkey[0] == 'parent':#if this is a parent parameter
                mask = self.free_mask[0]#handle frozen parent parameters
                # output = base_params[0]#get the full parent params
                #overwrite the full parent params with the values
                #from the deflated parameter while respecting
                #the frozen parameters.
                #first find if any masks sum to 0, this means an empty
                #parameter array would have been extracted, and subsequently
                #omitted during the deflation process. If this is the case
                #then insert an array of zeros in the shape of the mask in val.
                temp_val = []
                iv = 0
                for m in mask:#get the next val entry
                    if np.sum(m) == 0:
                        #then we know the parameter array at this index
                        #in vals was omitted. Don't increment the val array
                        #index. input empty array
                        val_entry = np.zeros(0)
                    else:
                        val_entry = val[iv]#otherwise get the next val_entry
                        iv += 1#increment the val_entry index
                    temp_val.append(val_entry)#store out the parameter
                val = lists_to_tuples(temp_val)
                for ip, mask in enumerate(mask):
                    # base_params[0][ip][mask][:] = val[ip]
                    if isinstance(val[ip], ArrayBox):
                        self.parent.params[ip][mask] = val[ip]._value
                    else:
                        self.parent.params[ip][mask] = val[ip]
                    
                # base_params[0] = reverse_apply_masks(mask, output, val)
            elif pkey[0] == 'lmbdas':
                #lambdas are not set up to be hierarchical or frozen
                #so just set the lambdas in the inflated parameter arrays
                #as the values of the deflated parameter
                # base_params[-1] = val
                if isinstance(val, ArrayBox):
                    self._sqrt_lmbdas = val._value[0]
                else:
                    self._sqrt_lmbdas = val[0]
            else:
                #if not parent or lmbdas, then it is a child parameter
                #and the pkey should be decomposed into (tag_tuple_entry, unique_value)
                #to update the child parameter values find all tags that have the same
                #unique value in the given tag tuple entry and update the child parameter.
                update_child_tags = [tag for tag in self.tags if tag[pkey[0]] == pkey[1]]
                #now get the hierarhcical param indices for this tag tuple entry
                #this will be a dictionary with keys corresponding to the
                #observation model parameter tuple indices and values that are
                #tuples of indices of the parameter array.
                hier_param_inds = self.hierarchical_params[pkey[0]]
                
                for tag in update_child_tags:
                    #find where the tag is in the base_param_labels
                    k = base_param_labels.index(tag)
                    #where to write the values to in the original child
                    #distribution parameter arrays. val will be a list
                    #of arrays denoting the subsets of the original
                    #parameter array for the child observation model.
                    #if multiple parameter types are hierarchical then
                    #there will be multiple entries in val. val will be
                    #packed in order of the hierarchical_params dictionary
                    #so we can unpack it in order as well.
                    for ic, (ip, pinds) in enumerate(hier_param_inds.items()):
                        # base_params[k][ip][pinds][:] = np.array(val[ic])
                        
                        #use to update the child parameter values directly
                        if isinstance(val[ic], ArrayBox):
                            self.children[tag].params[ip][pinds] = val[ic]._value
                        else:
                            self.children[tag].params[ip][pinds] = val[ic]
                    
                    #if there are some parameters in the observaiton models which
                    #are NOT hierarchically linked to the parent distribution parameters
                    #then set these equal to the parent parameters. This results in the
                    #model effectively only learning 1 distribution for these parameters,
                    #the parent distribution.
                    if self.not_hierarchical_mask != (None,):
                        for ip, pmask in self.not_hierarchical_mask.items():
                            # base_params[k][ip][pmask] = base_params[0][ip][pmask]#copy.deepcopy(base_params[0][ip][pmask])
                            
                            #use to update the child parameter values directly
                            self.children[tag].params[ip][pmask] = self.parent.params[ip][pmask]
        # #convert the lists back to tuples
        # base_params = lists_to_tuples(base_params)
        # return base_params
    
    def extract_free_params(self, params):
        # out = copy.deepcopy(params)
        for i, p in enumerate(params):
            params[i] = apply_masks(self.free_mask[i], p)
        
        return params
    
    def extract_frozen_params(self, params):
        # out = copy.deepcopy(params)
        for i, p in enumerate(params):
            inverse_mask = tuple(np.logical_not(arr) for arr in self.free_mask[i])
            params[i] = apply_masks(inverse_mask, p)
        return params
    
    def permute(self, perm):
        self.parent.permute(perm)
        for tag in self.tags:
            self.children[tag].permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        
        #need to respect frozen params, so get the values of the frozen params
        #before initializing the model
        if self.frozen != (None,):
            params_copy = self.params#copy.deepcopy(
            params_copy = tuples_to_lists(params_copy)
            fpvals = self.extract_frozen_params(params_copy)
            
        
        #set the lmdas
        # self._sqrt_lmbdas = self.set_lmbdas()
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags, init_method=init_method)
        
        if self.frozen != (None,):
            #now set the frozen parameters to be the stored values
            params_copy = self.params#copy.deepcopy(
            params_copy = tuples_to_lists(params_copy)
            
            for i, fpv in enumerate(fpvals):
                if len(fpv) > 0:
                    free_mask_ = self.free_mask[i]
                    inverse_mask = tuple(np.logical_not(arr) for arr in self.free_mask[i])
                    #set the frozen parameters to be the stored values
                    params_copy[i] = reverse_apply_masks(inverse_mask,  params_copy[i], fpv)
            #convert the lists back to tuples
            params_copy = lists_to_tuples(params_copy)
            self.params = params_copy
        
        # for tag in self.tags:
        #     if isinstance(self.lmbda, (np.ndarray, list, tuple)):
        #         self.children[tag].params = tuple(prm + np.sqrt(self.lmbda[i]) * npr.randn(*prm.shape) for i, prm in enumerate(self.parent.params))
        #     else:
        #         self.children[tag].params = tuple(prm + np.sqrt(self.lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)
            # self.children[tag].params = copy.deepcopy(self.parent.params)
        
        if self.hierarchical_params == (None,):
        #then there should be a lambda for each parameter distribution
            for tag in self.tags:
                for iprm, prm in enumerate(self.parent.params):
                    child_params = []
                    cprm = prm + self._sqrt_lmbdas[iprm] * npr.randn(*prm.shape)
                    child_params.append(cprm)
                self.children[tag].params = tuple(child_params)
        
        #if hierarchical params are specified, then this implies NOT all
        #child parameters are drawn from distributions around the parent params
        else:
            #first iterate through all child parameters and set them 
            #equal to the parent parameters. This will ensure that only
            #the parameters that are hierarchically linked to the parent
            #are drawn from distributions around the parent parameters.
            for tag in self.tags:
                self.children[tag].params = copy.deepcopy(self.parent.params)#
            
            #first extract the shared parameters
            ex_params = self.extract_shared_params()
            #need to iterate over the unpacked dictionary of hierarchical parameters
            #because the list index translates to the index of the lambda values.
            hier_tag_keys, hier_tag_vals = get_dict_leaf_items(self.hierarchical_params)
            for i, (ht_key, ht_vals) in enumerate(zip(hier_tag_keys, hier_tag_vals)):
                tag_ind = ht_key[0]#get the tag tuple entry index
                ppind = ht_key[1]#get the parent parameter list index
                #get the parent parameter values that are hierarchically linked
                prm = self.parent.params[ppind][ht_vals]
                #get the lambda value for this parameter distribution
                lambda_val = self._sqrt_lmbdas[i]
                #use the ex_params second layer keys which delineate
                #the unique values of the tag tuple entry
                #draw child parameters for each unique value for this tag ind
                for uval_key in ex_params[tag_ind].keys():
                    for tag in self.tags:#iterate over ALL tag labels 
                        #find where the tag tuple entry matches the unique value
                        if tag[tag_ind] == uval_key:
                            #draw child parameters around the parent parameters
                            cprm = prm + lambda_val * npr.randn(*prm.shape)
                            #parameters are tuples and cannot be modified in place.
                            #make a copy of the child parameters and update this
                            #mutable list copy. 
                            new_child_tuple = list(self.children[tag].params)#copy.deepcopy(self.children[tag].params)
                            #update the child parameter values at the appropriate
                            #parent param tuple index and parameter array indices
                            #from hierarchical_params
                            new_child_tuple[ppind][ht_vals] = cprm
                            #update the model's child parameter values
                            self.children[tag].params = tuple(new_child_tuple)


    def log_prior(self):
        lp = self.parent.log_prior()
        # if self.lambda_update == 'optimized':
            #add gaussian prior on the lambda params with strength sqrt(gamma)
            # lp += np.sum(norm.logpdf(self._sqrt_lmbdas, 0.025, np.sqrt(self.gamma)))
        # lambda_lower_bound = 5E-3#hiearchical prior lambda params cannot be negative
        # # # Clip the absolute values of lambdas to be larger than 1E-6
        # lmbdas = np.clip(self.lambdas, lambda_lower_bound, np.inf)
        # self._sqrt_lmbdas = np.sqrt(lmbdas)
        # Gaussian prior on sqrt lambdas
        # for cplmbda, lpmu in zip(self._sqrt_lmbdas, self.lmbda_prior_mu):
        #     lp += np.sum(norm.logpdf(cplmbda, 0.0, 0.1))

        # lmbdas = self.lambdas
        if self.hierarchical_params == (None,):
            # Gaussian likelihood on each child param given parent param
            for tag in self.tags:
                for pprm, cprm, cplmbda in zip(self.parent.params, self.children[tag].params, self._sqrt_lmbdas):
                    lp += np.sum(norm.logpdf(cprm, pprm, cplmbda))
        else:
            #first extract the shared parameters
            ex_params = self.extract_shared_params()
            #need to iterate over the unpacked dictionary of hierarchical parameters
            #because the list index translates to the index of the lambda values.
            hier_tag_keys, hier_tag_vals = get_dict_leaf_items(self.hierarchical_params)
            for i, (ht_key, ht_vals) in enumerate(zip(hier_tag_keys, hier_tag_vals)):
                tag_ind = ht_key[0]#get the tag tuple entry index
                ppind = ht_key[1]#get the parent parameter list index
                #get the parent parameter values that are hierarchically linked
                prm = self.parent.params[ppind][ht_vals]

                #use the ex_params second layer keys which delineate
                #the unique values of the tag tuple entry
                #draw child parameters for each unique value for this tag ind
                for uval_key in ex_params[tag_ind].keys():
                    for tag in self.tags:#iterate over ALL tag labels 
                        #find where the tag tuple entry matches the unique value
                        if tag[tag_ind] == uval_key:
                            #draw child parameters around the parent parameters
                            cprm = self.children[tag].params[ppind][ht_vals]
                            lp += np.sum(norm.logpdf(cprm, prm, self._sqrt_lmbdas[i]))
            
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # Optimize parent and child parameters at the same time with SGD
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):

            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)
                
                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        
        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, num_iters=num_iters, **kwargs)


class HierarchicalInitialStateDistribution(_Hierarchical):
    def log_initial_state_distn(self, data, input, mask, tag):
        return self.log_pi0 - logsumexp(self.log_pi0)


class HierarchicalTransitions(_Hierarchical):
    def log_transition_matrices(self, data, input, mask, tag):
        return self.children[tag].log_transition_matrices(data, input, mask, tag)


class HierarchicalObservations(_Hierarchical):
    def __init__(self, base_class, K, D, M, *args, tags=(None,), lmbda=0.01, frozen=(None,), hierarchical_params=(None,), gamma=0.0, lambda_update = 'fixed',**kwargs):
    # def __init__(self, base_class, K, D, M, *args, tags=(None,), lmbda=0.01, gamma=0.0, lambda_update = 'fixed', frozen=(None,), hierarchical_params=(None,),**kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda
        self.window_size = 5
        self.gamma = gamma
        self.lambda_update = lambda_update
        self.lmda_memory = []
        self.M = M
        self.K = K
        self.D = D
#         self.C = C


                # Variance of child params around parent params
        self.lmbda_prior_mu = lmbda
        self.lmbda = lmbda
        assert isinstance(self.lmbda, (float, int, list, tuple, np.ndarray)), "lmbda must be a float, list, tuple, or ndarray"
        if isinstance(self.lmbda, (float, int)):
            self.num_lambdas = 1
        else:
            self.num_lambdas = len(self.lmbda)
            
        # Top-level parameters (parent)
        self.parent = base_class(K, D, M, *args, **kwargs)
        # Set up child distributions
        self.children = dict()
        for tag in tags:
            self.children[tag] = base_class(K, D, M, *args, **kwargs)
            # ch = self.children[tag] = base_class(*args, **kwargs)
            # ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

        self._sqrt_lmbdas = self.set_lmbdas()
            
        # Make models for each tag
        self.tags = tags

        
        self.hierarchical_params = hierarchical_params
        if hierarchical_params != (None,):
            #hierarhcical params can be a list specifiying the indices of the parent parameter list
            #to draw the child parameter values around. If tags is a list of tuples,
            #then this indicates that multiple tag labels are used which should have
            #different hierarchical parameter sets. To specify these sets, the 
            #hierarchical_params should be a dictionary with first level keys 
            #that correspond to the tag tuple entries and the second level keys
            #should be ints that specify the parent parameter list indices and the leaf
            #values should be tuples that specify the parent parameter indices.
            assert isinstance(hierarchical_params, (list, dict)), "hierarchical_params must be a list or dictionary"
            
            #check if the entries of tags are tuples or lists they are all the same length
            #tags which are tuples indicate levels of similarity and difference between
            #child distributions. although each tuple denotes a different child distribution
            #if the entries of the tuples are the same, then the child distributions can be
            #made to share certain sets of underlying child distribution parameters. This 
            #effectively allows for imposing hierarchical structures on multiple specific 
            #parameters of the parent distribution. Before with only single tag labels 
            #all children parameters were distinct and only 1 factor could be used to impose
            #a hierarchical structure. now arbitrarily many factors can be used to impose 
            #hierarchical structure and the parameter subsets shared across tag/factor 
            #levels can be specified.
            if isinstance(tags[0], (tuple, list)):
                #check that all tags are tuples or lists
                assert all(isinstance(tags[i], (tuple, list)) for i in range(len(tags))) 
                if not all(len(tag) == len(tags[0]) for tag in tags):
                    raise ValueError("All tags must have the same number of entries")
                
                #now check that there are the same number of hierarchical parameters as
                #there are entries in the tags tuples. In this case this dictionary should 
                #have an integer key for each tag tuple entry and the value should be
                #a tuple of integers or a list of tuples of integers if separate parameter
                #indices for the same parameter type are to be controlled by unique lambdas.
                #The second level keys of hierarchical params should refer to unique list 
                #indices of the parent parameter list of arrays and thus are not larger 
                #than the length of the parent parameter array. 
                #The params property of the model is by definition a tuple of arrays, 
                #so if the hierarchy is to be established over specific axes of 
                #the parameter arrays, then the indices of these axes must be specified 
                #in the hierarchical_params dictionary with the second level dictionary keys 
                #specifying the parameter index of the parent parameter list and leaf values being tuple
                #entries or lists of tuple entries specifying the axes of the parameter array 
                #that are to be shared across the child distributions with the same tag tuple entry.
                ## Example dictionary to specify indices and slices:
                # index_slices = {
                #     0: {1: (0, 1, ...)},  # 2nd list entry, [0, 1, :]
                #     1: {0: (slice(1, 3), slice(2, 4)))}  # 1st list entry, [1:3, 2:4]
                # }
                assert isinstance(hierarchical_params, dict), "If tags are tuples, then hierarchical_params must be a dictionary"
                
                #set up a dictionary of masks that look just like the base
                #parameter arrays to explicitly map the shared parameters
                #this will be helpful because we will only optimize one copy
                #of the shared parameters but will need to update ALL the parameters
                #of the model.
                shared_mask_entry = []
                base_params = self.params
                #initialize the shared mask to be all ones
                # for i, mod in enumerate(base_params):
                #     temp_mask = []
                #     for mod_param in mod:
                #         temp_mask.append(np.zeros_like(mod_param))
                #     shared_mask_entry.append(temp_mask)
                
                # #for each tag tuple entry create a dictionary of shared parameter masks
                # #to indicate which parameters of the model are shared across the child
                # #distributions with the same tag tuple entry.
                # self.shared_mask = {k: copy.deepcopy(shared_mask_entry) for k in range(len(tags[0]))}

                if len(np.unique(list(self.hierarchical_params.keys()))) == len(tags[0]):
                    if not all(isinstance(self.hierarchical_params[k], dict) for k in self.hierarchical_params.keys()):
                        raise ValueError("All tag_params values must be dictionaries with keys \
                                        indicating parent parameter list indices and values that are tuples specifying parameter array indices")
                    
                    for k in list(self.hierarchical_params.keys()):
                        #check that all keys in the next level dictionary are integers
                        if not all(isinstance(k_, int) for k_ in self.hierarchical_params[k].keys()):
                            raise ValueError("If tag_params values are dictionaries, the keys must be integers")
                        #check that the maximum keys is less than the number of parent parameter list entries
                        if max(self.hierarchical_params[k].keys()) >= len(self.parent.params):
                            raise ValueError("The maximum key in the tag_params dictionary must be less than \
                                            the number of parent parameter list entries")
                    
                    #check that the specified hierarchical params indices and slices are valid
                    #i.e. that they are valid slices for the parameter arrays and that
                    #there is no overlap in the indices between the different tag tuple entries
                    shape_dict = {k: self.parent.params[k].shape for k in range(len(self.parent.params))}
                    result, message = check_index_overlap_multidimensional_with_shapes(self.hierarchical_params, shape_dict)
                    if not result:
                        raise ValueError(message)
                    
                else:
                    raise ValueError("The number of hierarchical parameters must \
                                        equal the number of entries in the tags tuples")
                #from hierarchical_params, find all indices that are NOT in the hierarchical_params
                self.not_hierarchical_mask = {}
                hier_leaf_keys, hier_leaf_values = get_dict_leaf_items(self.hierarchical_params)
                if self.num_lambdas != 1:
                    assert len(hier_leaf_keys) == len(self.lmbda), "Number of hierarchical parameters must match the number of lambdas, or use a single lambda for all parameters"
                
                for i in range(len(self.parent.params)):
                    #find any hierarchical_params values that refer to the given parameter index
                    #i.e. regardless of tuple entry (first layer keys) find all the leaf values
                    #behind 'i' in the second level keys of the hierarchical_params dictionary
                    leaf_values_i = [lv for lk, lv in zip(hier_leaf_keys, hier_leaf_values) if lk[1] == i]
                    
                    #if the parameter tuple index is NOT in the hierarchical_params
                    #then ALL the indices of the underlying parameter array are NOT
                    #hierarhcical.
                    if len(leaf_values_i) == 0:
                        self.not_hierarchical_mask[i] = np.ones_like(self.parent.params[i]).astype(bool)
                    #if instead the parameter tuple index is in the hierarchical_params
                    #then check if there are indices of the parameter array that are
                    #NOT in the hierarchical_params. If there are, then these indices
                    #are NOT hierarchical.
                    else:
                        #make a mask to collect where the indices
                        #in the hierarchical_params are NOT in the parameter array
                        total_mask = np.ones_like(self.parent.params[i])
                        for lv in leaf_values_i:
                            #set the mask to zero for the hierarchical parameters
                            total_mask[lv] = 0
                        
                        self.not_hierarchical_mask[i] = total_mask.astype(bool)
            
        #list of parameters that are free to be optimized
        #this must be defined at the resolution of the 
        #parameter getter method output which is a tuple of
        #tuples of parameter arrays
        self.frozen = frozen
        #initialize a mask to translate the frozen parameter
        #slices into a mask for each parameter array. This will
        #facilitate updating/setting the parameters during optimization.
        self.free_mask = []
        base_params = self.params
        #initialize the free mask to be all ones
        for i, mod in enumerate(base_params):
            fmask_entry = []
            if isinstance(mod, (list, tuple)):
                for mod_param in mod:
                    fmask_entry.append(np.ones_like(mod_param).astype(bool))
            elif isinstance(mod, np.ndarray):
                fmask_entry.append(np.ones_like(mod).astype(bool))
            else:
                raise ValueError("Model parameters must be a tuples of arrays or arrays")

            self.free_mask.append(fmask_entry)
                
        if self.frozen != (None,):#then assume all parameters are free
            assert isinstance(self.frozen, (list, tuple, dict)), "frozen must be a list, tuple, or dict"
            if isinstance(self.frozen, (list, tuple)):
                assert all(isinstance(f, int) for f in self.frozen), "free must be a list of integers specifying indices in self.params"
                assert max(self.frozen) < len(base_params), "free index out of range of self.params. see getter method for self.params"
                for f in self.frozen:
                    self.free_mask[f] = np.zeros_like(base_params[f]).astype(bool)
                #check that none of the frozen parameters are hierarchical parameters
                #this will only happen if parent paremeters are frozen, i.e.
                #0 is in the frozen list
                if 0 in self.frozen:
                    for tag_key in self.hierarchical_params:
                        if 0 in self.hierarchical_params[tag_key].keys():
                            raise ValueError("Hierarchical parameters cannot be frozen")
            else:#if frozen is a dictionary
                assert dict_depth(self.frozen) <= 2, "frozen dictionary must have depth <= 2, first level hierarchical param tuple, second level specific child model param tuples."
                assert all(isinstance(f, int) for f in self.frozen.keys()), "frozen keys must be integers specifying indices in the base params tuple, see params property"
                assert max(self.frozen.keys()) < len(base_params), "frozen key out of range of the base params tuple, see params property"
                leaf_keys, leaf_values = get_dict_leaf_items(self.frozen)
                #check that all leaf values are lists of tuples, or None
                assert all(isinstance(f, (list, tuple, type(None))) for f in leaf_values), "frozen leaf values must be lists of tuples or None"
                #now check that each tuple index is valid
                for i, leaf in enumerate(leaf_values):
                    if leaf is not None:
                        if isinstance(leaf, list):
                            for tup in leaf:
                                #get the model parameter array
                                param_array = get_nested_tuple_value(base_params, leaf_keys[i])
                                if not is_valid_slice_for_shape(tup, param_array):
                                    raise ValueError("Invalid slice for array")
                                else:
                                    #set the mask to zero for the frozen parameter
                                    set_val = np.ones_like(param_array).astype(bool)
                                    set_val[tup] = 0
                                    set_nested_list_value(self.free_mask, leaf_keys[i], set_val)
                                    
                        else:
                            tup = leaf#leaf is a tuple indexing the parameter array
                            #get the model parameter array
                            param_array = get_nested_tuple_value(base_params, leaf_keys[i])
                            if not is_valid_slice_for_shape(leaf, param_array.shape):
                                raise ValueError("Invalid slice for array")
                            else:
                                #set the mask to zero for the frozen parameter
                                set_val = np.ones_like(param_array).astype(bool)
                                set_val[tup] = 0
                                set_nested_list_value(self.free_mask, leaf_keys[i], set_val)
                    
                    else:#if the leaf value is None, then it means that ALL the parameter indices at i are frozen
                        param_array = get_nested_tuple_value(base_params, leaf_keys[i])
                        #set the mask to zero for the frozen parameter
                        set_val = np.zeros_like(param_array).astype(bool)
                        set_nested_list_value(self.free_mask, leaf_keys[i], set_val)
                        
                #check that none of the frozen parameters are hierarchical parameters
                #this will only happen if parent paremeters are frozen, i.e.
                # #0 is in the first level of the frozen dictionary
                # if 0 in self.frozen.keys():
                #     for i, inds in enumerate(self.frozen[0].items()):
                #         for tag_key in self.hierarchical_params:
                #             #get all terminal values behind key i
                #             leaf_keys, leaf_values = get_dict_leaf_items(inds)
                #             leaf_values = [lv for lk, lv in zip(leaf_keys, leaf_values) if lk == i]
                #             #flatten the list
                #             leaf_values = [item for sublist in leaf_values for item in sublist]
                        
                #         if leaf_values:#if there are any leaf values
                #             #check if there is overlap between the frozen parameter
                #             #indices and the hierarchical parameter indices for 
                #             #this parameter
                #             pshape = self.parent.params[i].shape
                #             result, message = check_overlap_between_tuples_or_lists(inds, leaf_values, pshape)
                #             if result:
                #                 raise ValueError(message)
                        
        #convert the free mask to a tuple of tuples to make it immutable
        self.free_mask = convert_nested_list_to_tuple(self.free_mask)   
        
        # self.set_lmdas()
        # self.initialize()

    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag, with_noise=with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(expectations, data, input, tag)

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # expected log joint
        def _expected_log_joint(expectations):
            if (self.lambda_update in ['recursive', 'optimized']):
                self.lmda_memory.append(self._sqrt_lmbdas)
            if self.lambda_update == 'recursive':
                self.update_lmbdas()

            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params_, itr):
            #set the parameters of the model to the passed param values
            #prior to calculating the expected log joint
            if self.hierarchical_params == (None,):
                self.params = params_
            else:
                #params will be deflated, i.e. only the hierarchical and free params
                #need to inflate the params to the full parameter set to set
                #the model parameters
                self.inflate_params(params_)
                
            # if self.lambda_update == 'optimized':
            #     prior_param_pen = np.linalg.norm(np.array(self.params[-1]), ord=2)**2
            # else:
            #     prior_param_pen = 0.
            obj = _expected_log_joint(expectations)# - self.gamma*prior_param_pen
            return -obj / T

        # self.params = \
        #     optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)
        
        # self.params = \
        #     adamc(grad(_objective), self.params, num_iters=num_iters, **kwargs)

        lambda_lower_bound = 1E-6#hiearchical prior lambda params cannot be negative
        lambda_upper_bound = 1E-2
        # var_lower_bound = 1E-6#variance params cannot be negative
        #make bounds so that the None is returned for all params except the lambda params, the last entry in the params tuple
        # low_bounds = []
        # up_bounds = []
        # param_count = len(self.params)
        # if self.lambda_update == 'optimized':
        #     param_count -= 1
        
        # if optimizer == 'lbfgs':
        # #     for i in range(param_count):#for each param tuple.
        # #         lb_tup = []
        # #         ub_tup = []
        # #         for j in range(len(self.params[i])):#for each param type in the tuple
        # #             # if j == len(self.params[i])-1:#the variance param is the last entry in the tuple
        # #             #     lb_tup.append(np.full_like(self.params[i][j], var_lower_bound))
        # #             #     ub_tup.append(np.full_like(self.params[i][j], np.inf))
        # #             # else:#all other params are bounded by +/- infinity
        # #             lb_tup.append(np.full_like(self.params[i][j], -np.inf))
        # #             ub_tup.append(np.full_like(self.params[i][j], np.inf))

        # #         lb_tup = tuple(lb_tup)
        # #         ub_tup = tuple(ub_tup)
        # #         low_bounds.append(lb_tup)
        # #         up_bounds.append(ub_tup)

        # #     if self.lambda_update == 'optimized':
        # #         # #collect the parameter arrays as stacked arrays for each parameter type in temp
        # #         # temp = self.stack_child_params()
        # #         # #iterate over each parameter vector and calculate the standard deviation between the states 
        # #         # # of the average parameter vectors across all child distributions.
        # #         # btwn_state_stds = [0]*len(self.parent.params)
        # #         # for j, spvecs in enumerate(temp):#for each parameter vector in the child params set
        # #         #     #average across each parameter vector and then take the standard deviation
        # #         #     btwn_state_stds[j] = np.std(np.mean(spvecs, axis=-1))
        # #         #     # #ensure that the standard deviation is larger than the lower bound..
        # #         #     if btwn_state_stds[j] < lambda_lower_bound:
        # #         #         btwn_state_stds[j] = lambda_lower_bound + lambda_lower_bound/2
        # #         #     elif btwn_state_stds[j] > lambda_upper_bound:#truncate at 
        # #         #         btwn_state_stds[j] = lambda_upper_bound
        # #         #the last param is always the lambda params, one value for each observation model parameter
        # #         low_bounds.append(tuple([np.full_like(self.params[-1][j], lambda_lower_bound) for j in range(len(self.params[-1]))]))#add the bounds for the lambda params
        # #         up_bounds.append(tuple([np.full_like(self.params[-1][j], lambda_upper_bound) for j in range(len(self.params[-1]))]))#add the bounds for the lambda params, btwn_state_stds[j]
        #     #package into a single bounds tuple
        #     # bounds = tuple(zip(low_bounds, up_bounds))
        #     bounds = None
        #     #construct var_lower_bound array in the same shape as the variance params array
        #     if self.hierarchical_params == (None,):
        #         self.params = \
        #             lbfgs(_objective, self.params, bounds=bounds, num_iters=num_iters, **kwargs)
        #     else:
        #         #deflate the parameters to only the free and hierarchical params
        #         prms = self.deflate_params()
        #         new_params = \
        #             lbfgs(_objective, prms, bounds=bounds, num_iters=num_iters, **kwargs)
        #         #inflate the new parameters to the full parameter set
        #         new_full_params = self.inflate_params(new_params)
        #         #call params setter method now
        #         self.params = new_full_params
        # else:
        # Optimize parent and child parameters at the same time with SGD
        # optimizer = dict(adam=adam, sgd=sgd)[optimizer]# bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop,
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]


        if self.hierarchical_params == (None,):
            self.params = \
                optimizer(_objective, self.params, num_iters=num_iters, **kwargs)
        else:
            #deflate the parameters to only the free and hierarchical params
            prms = self.deflate_params()
            # prms = optimizer(grad(_objective), prms, num_iters=num_iters, **kwargs)
            prms = optimizer(_objective, prms, num_iters=num_iters, **kwargs)
            #inflate the new parameters to the full parameter set
            #call params setter method now
            self.inflate_params(prms)


class HierarchicalEmissions(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_y(self, z, x, input=None, tag=None):
        return self.children[tag].sample_y(z, x, input=input, tag=tag)

    def initialize_variational_params(self, data, input, mask, tag):
        return self.children[tag].initialize_variational_params(data, input, mask, tag)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        return self.children[tag].smooth(expected_states, variational_mean, data, input, mask, tag)

