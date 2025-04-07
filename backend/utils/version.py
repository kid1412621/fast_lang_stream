def compare_versions(version1, version2):
    """
    Compare two version strings.
    
    Args:
        version1 (str): First version string (e.g., "1.2.3")
        version2 (str): Second version string (e.g., "1.3.0")
        
    Returns:
        int: 1 if version1 > version2
             0 if version1 == version2
            -1 if version1 < version2
    """
    v1_parts = [int(x) for x in version1.split('.')]
    v2_parts = [int(x) for x in version2.split('.')]
    
    # Make both lists the same length by padding with zeros
    max_length = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_length - len(v1_parts)))
    v2_parts.extend([0] * (max_length - len(v2_parts)))
    
    # Compare each component
    for i in range(max_length):
        if v1_parts[i] > v2_parts[i]:
            return 1
        elif v1_parts[i] < v2_parts[i]:
            return -1
            
    # If we get here, the versions are equal
    return 0

def is_version_at_least(version, min_version):
    """
    Check if version is greater than or equal to min_version
    """
    return compare_versions(version, min_version) >= 0