List<String> listDifferences(List<dynamic> list1, List<dynamic> list2) {
  List<String> differences = [];

  if (list1.length != list2.length) {
    differences.add(
        'Lists have different lengths: ${list1.length} != ${list2.length}');
    return differences;
  }

  for (int i = 0; i < list1.length; i++) {
    dynamic item1 = list1[i];
    dynamic item2 = list2[i];

    if (item1 is Map && item2 is Map) {
      List<String> mapDiff = mapDifferences(
        item1 as Map<String, dynamic>,
        item2 as Map<String, dynamic>,
      );
      if (mapDiff.isNotEmpty) {
        differences.add('Differences in map at index $i:');
        differences.addAll(mapDiff);
      }
    } else if (item1 is double && item2 is double) {
      if (item1.toStringAsPrecision(1) != item2.toStringAsPrecision(1)) {
        differences.add('Value mismatch at index $i: $item1 != $item2');
      }
    } else if (item1 is List && item2 is List) {
      List<String> listDiff = listDifferences(item1, item2);
      if (listDiff.isNotEmpty) {
        differences.add('Differences in list at index $i:');
        differences.addAll(listDiff);
      }
    } else {
      if (item1 != item2) {
        differences.add('Value mismatch at index $i: $item1 != $item2');
      }
    }
  }

  return differences;
}

List<String> mapDifferences(
    Map<String, dynamic> map1, Map<String, dynamic> map2) {
  List<String> differences = [];

  if (map1.keys.length != map2.keys.length) {
    differences.add('Maps have different number of keys.');
    return differences;
  }

  for (String k in map1.keys) {
    if (!map2.containsKey(k)) {
      differences.add('Key $k is missing in second map.');
      continue;
    }

    dynamic val1 = map1[k];
    dynamic val2 = map2[k];

    if (val1 is Map && val2 is Map) {
      List<String> nestedDifferences = mapDifferences(
          val1 as Map<String, dynamic>, val2 as Map<String, dynamic>);
      if (nestedDifferences.isNotEmpty) {
        differences.add('Differences in nested map for key $k:');
        differences.addAll(nestedDifferences);
      }
    } else if (val1 is double && val2 is double) {
      if (val1.toStringAsPrecision(1) != val2.toStringAsPrecision(1)) {
        differences.add('Value mismatch for key $k: $val1 != $val2');
      }
    } else if (val1 is List && val2 is List) {
      List<String> listOfDifferences = listDifferences(val1, val2);
      if (listOfDifferences.isNotEmpty) {
        differences.add('Differences in list for key $k:');
        differences.addAll(listOfDifferences);
      }
    } else {
      if (val1 != val2) {
        differences.add('Value mismatch for key $k: $val1 != $val2');
      }
    }
  }

  return differences;
}

bool listEquals(List<dynamic> list1, List<dynamic> list2) {
  List<String> differences = listDifferences(list1, list2);
  return differences.isEmpty;
}
