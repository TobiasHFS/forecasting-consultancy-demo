# Zwei Prognose-Demos: Methodik und Grenzen

Diese Demos prüfen eine vollständige Kette: **Informationen am Entscheidungszeitpunkt → Prognose → Mengenentscheidung → simuliertes wirtschaftliches Ergebnis**. Ein besserer Prognosefehler ist ein Zwischenresultat. Für den praktischen Nutzen zählt, ob ein Betrieb aufgrund der Prognose besser handeln kann.

Alle Demo-Beobachtungen sind synthetisch. Generator, Zufallsstartwerte, Kostenannahmen und Modellentscheidungen sind im Projekt nachprüfbar. Die tatsächlichen Ergebnisse werden beim Ausführen erzeugt und im Bericht aus den gespeicherten Dateien übernommen. Hier werden bewusst keine noch nicht berechneten Zahlen eingesetzt.

## 1. Welche Aussage ein erfolgreicher Lauf erlaubt

Ein guter Lauf zeigt, dass die technische und methodische Kette unter den dokumentierten Simulationsannahmen funktioniert und gegenüber den getesteten einfachen Alternativen einen Unterschied erzeugt. Er zeigt nicht, dass ein konkreter Kunde entsprechend viel sparen würde. Synthetische Daten können die Strukturen enthalten, die dem ausgewählten Modell entgegenkommen. Daher müssen auch starke Baselines, andere Zufallsstartwerte, alternative Kosten und veränderte Bedingungen betrachtet werden.

Die Demos sind **prädiktiv und entscheidungsorientiert**, nicht kausal identifizierte Kundenstudien. Wenn mehrere Regeln auf dieselbe exogen gehaltene simulierte Nachfrage angewendet werden, ist der Unterschied ein Simulationsvergleich. Er ist kein randomisierter Wirkungsnachweis im Geschäftsbetrieb und kein Preiselastizitätsnachweis. Preisaktionen können als bekannte erklärende Variable vorkommen; ihre Regressionskoeffizienten oder Feature-Importances sind dadurch noch keine kausalen Preiseffekte.

## 2. Zwei Entscheidungen mit unterschiedlichen Horizonten

| Entscheidung | Zu prognostizieren | Typischer Entscheidungszeitpunkt | Wichtige Randbedingungen |
|---|---|---|---|
| Tagesmenge verderblicher Ware | Nachfrage bzw. Restnachfrage für den nächsten Verkaufstag, nach klarer Artikel-/Standortabgrenzung | Vor Produktion oder Bestellung; spätere Verkäufe noch unbekannt | Variable Stückkosten, Verkaufs-/Restwert, Über- und Untermenge, Kapazität, Rundung und Frische |
| Wiederbeschaffung lagerfähiger Artikel | Nachfrage über Lieferzeit **plus** Zeit bis zur nächsten Bestandsprüfung | Beim festgelegten Bestellstichtag | Verfügbarer Bestand, Pipeline/offene Bestellungen, Lieferzeit, Mindestgebinde, Bestellrhythmus, Lager- und Fehlmengenkosten |

Die konkreten Namen, Datenperioden, Artikelzahlen und Horizonte eines ausgeführten Falls stehen in seinen Ausgabedateien und im Bericht. Sie werden nicht als Kundendaten ausgegeben. Bekannte feste Aufträge gehören direkt in die Planung; nur der am Stichtag unbekannte Teil braucht eine Nachfrageprognose.

## 3. Synthetische Nachfrage ist nicht beobachteter Verkauf

Sei `D_t` die tatsächliche Nachfrage und `A_t` die verfügbare Menge. In einem einfachen Lost-Sales-Modell gilt:

`Verkauf_t = min(D_t, A_t)`

`Fehlmenge_t = max(D_t − A_t, 0)`

`Restmenge_t = max(A_t − D_t, 0)`

Bei Ausverkauf kennt ein echter Betrieb den nicht erfüllten Bedarf gewöhnlich nicht vollständig. Ein Modell auf Kassendaten kann dann Bestandsknappheit als niedrige Nachfrage lernen. Auch Bestandsfehler sind kritisch. [Q08](QUELLEN.md)

Der Generator speichert die latente Nachfrage in einem getrennten Auditbereich, damit sich Regeln im Simulator auf demselben Pfad vergleichen lassen. **Sie wird in diesem Projekt nicht als Trainingsziel, Merkmal, Auswahlkriterium oder Kalibrierungseingang verwendet.** Die vollständige Kenntnis ist ausschließlich eine Simulator-Referenz für die nachgelagerte Bewertung. Die Modelle erhalten beobachtete Verkäufe und die dokumentierten Verfügbarkeits-/Ausverkaufsinformationen. Ein echter Pilot müsste entsprechend die Zensierung modellieren oder zusätzliche Nachfragedaten erheben; eine bloße Gewichtung knapper Tage rekonstruiert keine vollständige Nachfrage.

In der getrennten Bäckerei-Replikation wird eine rechtszensierte Lognormalregression untersucht. Für einen nicht ausverkauften Tag trägt die beobachtete Menge zur Dichte bei; bei Ausverkauf ist nur bekannt, dass Nachfrage mindestens die verfügbare Menge erreicht. Dann trägt die Überlebenswahrscheinlichkeit `P(D ≥ verfügbare Menge | x)` zur Likelihood bei. Eine Punktregression auf die Verkaufsschranke würde diese Information falsch als vollständigen Zielwert behandeln. Die Implementierung modelliert `D + 0,5` mit einer kontinuierlichen Lognormalverteilung und artikelspezifischer Log-Streuung. Dies ist eine Näherung für diskrete Stückzahlen. Regularisierung wird anhand zensierter Validierungs-Likelihood und der spätere Streuungsmultiplikator anhand separater zensierter Kalibrierungs-Likelihood gewählt, nicht anhand versteckter Nachfrage oder späterer Euro-Werte. [Q10, Q11](QUELLEN.md)

Die gewählte Lognormalform ist eine zusätzliche Modellannahme. Nötig sind zuverlässige Verfügbarkeitsschranken, ein passender bedingter Nachfrageverlauf und ein Produktions-/Zensierungsmechanismus, der bedingt auf bekannte Kalender-, Wetter- und historische Merkmale keine zusätzliche unbekannte aktuelle Nachfrageinformation enthält. Diese Annahme kann in einem echten Betrieb verletzt sein, etwa wenn erfahrene Mitarbeiter morgens zusätzliche Signale nutzen, die im Export fehlen. Die Regression garantiert keine Rekonstruktion jedes verlorenen Verkaufs. Das günstige Wissen über plausible Interaktionen im synthetischen Design begrenzt die Übertragbarkeit zusätzlich.

Nullnachfrage, Ladenschluss, Ausverkauf und fehlender Datenexport sind verschiedene Zustände. Fehlende Beobachtungen werden nicht ohne Prüfung zu Nullverkäufen umgewandelt. Jeder Fall benötigt stabile Artikel-/Standortkennungen, Zeitstempel und einen nachvollziehbaren Kalender.

## 4. Zeitliche Trennung hält Zukunftsinformationen fern

Ein zufälliger Zeilensplit kann scheinbar gute Prognosen erzeugen, weil spätere Informationen oder ähnliche Tage bereits im Training vorkommen. Das Verfahren folgt deshalb einem chronologischen Vertrag. [Q01, Q03](QUELLEN.md)

1. **Training:** frühere Daten zur Parameterschätzung.
2. **Zeitliche Validierung/Modellwahl:** spätere, im Training nicht enthaltene Prognoseursprünge; gegebenenfalls mehrere rollierende Folds.
3. **Kalibrierung:** sofern genutzt, ein separat begründetes späteres Segment für Intervallkorrekturen. Es darf nicht unbemerkt zur erneuten Modellwahl dienen.
4. **Finaler Test:** der zuletzt liegende Zeitabschnitt; keine Hyperparameter- oder Mengenregelwahl anhand seiner Ergebnisse.

Bei jeder Prognose sind ausschließlich Merkmale zulässig, die am damaligen Stichtag verfügbar waren. Ein Lag `y_(t−1)` ist erlaubt, wenn der entsprechende Wert bereits feststand. Rollierende Mittel müssen entsprechend verschoben sein. Tatsächliches Wetter des morgigen Tages, später eingegangene Bestellungen, nachträglich korrigierte Lieferzusagen oder die endgültige Ausverkaufsdauer sind keine zulässigen Eingänge für eine frühere Entscheidung.

Mehrschrittziele können Beobachtungen über mehrere Tage summieren. Wenn Trainingsziele in ein späteres Validierungsfenster hineinreichen, entsteht ebenfalls Informationsüberschneidung. Die Fold-Grenze muss deshalb den Zielhorizont berücksichtigen, beispielsweise mit einer passenden Lücke. Der Bericht muss erklären, ob es um echte mehrtägige Prognosen am ursprünglichen Stichtag oder tägliche Aktualisierungen geht.

## 5. Erst einfache Baselines, dann das ML-Modell

Baseline-Prognosen sind Teil des Tests: letzte vergleichbare Wochentage, rollierende Mittel, einfache saisonale Regeln und gegebenenfalls ein sparsames lineares/statistisches Modell. Sie bekommen denselben Informationsstand und denselben Prognosehorizont wie das ML-Modell. Eine künstlich schwache Baseline würde die Demo attraktiver, aber weniger glaubwürdig machen.

Histogramm-basiertes Gradient Boosting baut schrittweise Entscheidungsbäume, die verbleibende Fehler korrigieren. Es kann nichtlineare Kalender-, Wetter-, Artikel- und Lag-Beziehungen kombinieren. Das ist nützlich, wenn echte wiederkehrende Muster existieren; es garantiert weder Überlegenheit noch Extrapolation in neue Geschäftsbedingungen. Anzahl der Bäume, Lernrate, Blattgröße und Regularisierung werden sparsam und innerhalb der zeitlichen Validierung gewählt. [Q02](QUELLEN.md)

Die automatische interne Validierung des Schätzers ist kein Ersatz für den beschriebenen Zeitplan. Wenn der verwendete Ablauf keine ausdrücklich chronologische interne Validierung bereitstellt, wird internes zufälliges Early Stopping deaktiviert und die Modellwahl extern zeitlich geprüft. Bibliotheksversion, Merkmalsliste, Kandidaten und gewählte Parameter gehören ins Auswahlprotokoll.

Ein gemeinsames Modell über mehrere Artikel kann Informationen teilen. Es muss aber aussagekräftige Kennungen/Merkmale enthalten und auf späteren Daten sowie einzelnen Artikelgruppen geprüft werden. Ein guter Gesamtwert kann schlechte Ergebnisse für kleine oder teure Artikel verdecken. Feature-Importances beschreiben modellinterne Zusammenhänge und werden nicht als Wirkung einer Preis-, Wetter- oder Bestandsänderung interpretiert.

## 6. Punktprognose und Unsicherheit beantworten verschiedene Fragen

Eine Punktprognose schätzt etwa die mittlere oder mediane Nachfrage. Ein Quantil `q_τ(x)` beschreibt eine Grenze, unter der ungefähr der Anteil `τ` der bedingten Nachfrage liegen soll. Bei `loss="quantile"` schätzt das verwendete Boosting-Verfahren die angegebene Quantilstufe anhand des Pinball-Verlusts. [Q02, Q04](QUELLEN.md)

`Pinball_τ(y,q) = τ × (y−q), wenn y ≥ q; sonst (1−τ) × (q−y)`

Getrennt geschätzte Quantile können sich kreuzen. Ein gegebenenfalls verwendetes Sortieren oder eine andere Monotoniekorrektur ist zu protokollieren; es ist kein zusätzlicher Genauigkeitsnachweis. Empirische Abdeckung und Breite werden auf späteren Daten gemeinsam geprüft. Ein sehr breites Band kann fast alles abdecken und trotzdem nutzlos sein.

Eine Residuen- oder Conformal-Kalibrierung kann empirische Fehlkalibrierung reduzieren. Die bekannte endliche Stichprobengarantie klassischer Conformal-Verfahren beruht jedoch auf Austauschbarkeit. Zeitliche Abhängigkeit, Saisonalität oder Drift können sie verletzen. Die Existenz spezieller Verfahren für abhängige Daten überträgt deren Theorie nicht automatisch auf unsere Implementierung. Der Bericht beschreibt deshalb **empirische Testabdeckung**, keine voraussetzungslose Garantie für zukünftige Tage. [Q05](QUELLEN.md)

Für Bestandsplanung gilt außerdem:

`Quantil(Summe täglicher Nachfrage) ≠ Summe täglicher Quantile` im Allgemeinen.

Die Schutzperiodennachfrage muss direkt modelliert oder aus gemeinsamen Nachfragepfaden simuliert werden. Tagesabhängigkeit darf dabei nicht stillschweigend entfernt werden. Ein tägliches 90%-Quantil ist kein 90%-Quantil einer mehrtägigen Summe.

## 7. Prognosemetriken und fairer Modellvergleich

| Maß | Bedeutung | Grenze |
|---|---|---|
| MAE | Mittlere absolute Abweichung in Stück, kg oder der jeweiligen Zielgröße | Gewichtet Fehlrichtungen symmetrisch; unterschiedliche Mengenmaßstäbe beachten |
| RMSE | Bestraft große Fehler stärker | Wenige Extremtage können den Vergleich dominieren |
| WAPE, falls ausgewiesen | Summe absoluter Fehler geteilt durch gesamte beobachtete Menge | Mengenstarke Reihen dominieren; bei Null-Gesamtmenge nicht definiert |
| Pinball-Verlust | Güte der jeweiligen Quantilsprognose | Quantilstufe und wirtschaftliche Bedeutung angeben |
| Empirische Intervallabdeckung | Anteil späterer Beobachtungen im vorhergesagten Band | Zusammen mit Breite und Gruppen-/Zeitstabilität beurteilen |
| Entscheidungskennzahlen | Fehl-/Restmengen, Lieferfähigkeit, Lagerbindung und simulierte Kosten | Gelten nur unter den Annahmen der simulierten Regel |

Ein Modell wird über Validierungsdaten ausgewählt. Der finale Test kann zeigen, dass die Auswahl dort nicht gewinnt. Dieses Resultat bleibt sichtbar. Es wird kein Testmodell nachträglich als vorher ausgewählten Sieger bezeichnet. Falls erneut trainiert wird, müssen Trainingsende und sämtliche weiterhin unberührten Testgrenzen im Protokoll eindeutig bleiben.

## 8. Verderbliche Ware: die Kosten bestimmen die Mengenregel

Im einfachen einperiodigen Newsvendor-Modell sind `c_u` die Kosten einer fehlenden Einheit und `c_o` die Kosten einer überschüssigen Einheit. Bei positiver Summe liegt das kostenorientierte Zielquantil bei:

`τ* = c_u / (c_u + c_o)`

Bei Verkaufspreis `p`, variabler Stückkostenannahme `c` und Restwert `s` ist ein einfacher Spezialfall `c_u = p−c` und `c_o = c−s`. Das gilt nur für diesen Modellrahmen. Eine Fehlmengeneinheit kostet nicht automatisch den ganzen Umsatz; fixe Personalkosten verschwinden nicht automatisch, wenn weniger produziert wird. [Q06](QUELLEN.md)

Die Mengenregel muss danach Ganzzahligkeit, Mindestmengen und gegebenenfalls Kapazität beachten. Ein knappes gemeinsames Produktionslimit macht unabhängige Artikelentscheidungen nicht automatisch global optimal. Der Code muss offenlegen, wie Mengen bei Kapazitätsknappheit zugeteilt werden.

Ein glaubwürdiger Vergleich trennt zwei mögliche Beiträge: bessere Prognosen und bessere Mengenregeln. Soweit umgesetzt, sollte dieselbe kostenorientierte Regel mit einer einfachen Prognose und mit der ML-Prognose verglichen werden. Der reine Wechsel von einer heutigen Faustregel zu einer besseren Entscheidungsregel darf nicht vollständig als ML-Effekt ausgegeben werden.

## 9. Lagerartikel: Pipeline und Endbestand gehören in die Rechnung

Für eine periodische Order-up-to-Regel ist die Schutzperiode die Lieferzeit `L` plus das Prüfintervall `R`. Die Bestandsposition umfasst verfügbaren Bestand und offene Bestellungen, vermindert um relevante Verpflichtungen. Eine schematische Regel lautet:

`Bestellung = max(0, Zielbestand − Bestandsposition)`

Rundung, Mindestgebinde, Lieferkalender und tatsächliche Ankunft werden anschließend im Simulator angewendet. Der Schutzperiodenbedarf umfasst bereits bekannte Bedarfe plus den prognostizierten unbekannten Teil. Feste Aufträge werden nicht doppelt gezählt. [Q07](QUELLEN.md)

Fehlmengen können verlorene Verkäufe oder spätere Rückstände sein. Das sind verschiedene Geschäftsmodelle; die Demo muss eines explizit verwenden. Ein Ergebnis für Lost Sales ist nicht ohne Anpassung auf Backorders übertragbar.

Eine endliche Simulation darf den Endbestand nicht gratis verschwinden lassen oder ankommende Pipeline ignorieren. Initialbestand, Warm-up, Endbestand und offene Bestellungen sind zu dokumentieren und zwischen Regeln fair zu behandeln. Getrennt zu zeigen sind Kapitalbindung und Kosten: freigesetzter Warenwert ist kein Gewinn in gleicher Höhe. Wenn ein Haltekostensatz bereits Kapital- und Lagerkosten enthält, werden sie nicht nochmals addiert.

## 10. Wirtschaftliche Ergebnisse sind Szenarien

Die Demos vergleichen Regeln auf gleichen simulierten Nachfragepfaden und möglichst gleichen exogenen Liefer-/Kostenbedingungen. Ein Vorteil kann nur für die dokumentierte Testperiode, Artikelmenge, Regel und Kostenkonfiguration beziffert werden. Modellkosten, Betrieb, Integration und Kundenschulung sind nicht automatisch enthalten. Ohne diese Kosten ist das Ergebnis kein vollständiger Anbieter-/Kunden-ROI.

Kosten- und Lieferzeitannahmen gehören offen in eine Parametertabelle. Zwei unterschiedliche Sensitivitäten werden klar getrennt: **feste Mengen neu bewerten** und **unter neuen Bedingungen neu planen/simulieren**. Die Bäckerei-Datei `frozen_quantity_cost_stress` bewertet die eingefrorenen Mengen mit anderen Kosten gewichtet; sie zeigt die Empfindlichkeit dieses Plans, aber keine kostenoptimal neu berechnete Mengenregel. Änderungen von Lieferzeit, Gebinden oder kostenabhängigen Zielmengen erfordern einen vollständigen Neureplay. Eine proportionale Hochrechnung würde dann Bestellentscheidungen, Lieferfähigkeit und Endbestand falsch abbilden.

Wird ein Ergebnis annualisiert, ist es lediglich eine arithmetische Illustration unter unveränderten Bedingungen. Ferien, Saison, Wachstums-/Sortimentswechsel oder abweichende Kundengröße sind darin nicht validiert. Im Kundengespräch sollte der beobachtete Pilotzeitraum im Vordergrund bleiben.

## 11. Bootstrap und mehrere Zufallsstartwerte haben begrenzte Aussagen

Ein gepaarter Vergleich verwendet dieselben Tage und Artikel für beide Regeln. Tages-/Wochenblöcke können die gemeinsame Zeitabhängigkeit besser erhalten als unabhängig gezogene Einzelzeilen. Blocklänge, Resampling-Einheit, Zahl der Wiederholungen und Kennzahl sind offen zu protokollieren. Zeitreihen-Bootstrap setzt geeignete Stabilität/Abhängigkeit voraus. [Q09](QUELLEN.md)

Das resultierende Band beschreibt Variabilität **innerhalb** des verwendeten Simulations-/Resamplingrahmens. Es berücksichtigt nicht automatisch unbekannte Kundendatenqualität, Marktwechsel, misspezifizierte Nachfrage, unerkannte Ausverkäufe oder falsche Kosten. Mehr Bootstrap-Wiederholungen erzeugen keine neuen echten Kundenbeobachtungen.

Weitere Generator-Startwerte prüfen, ob ein Ergebnis vom konkreten Zufallspfad abhängt. Alternative Generatorstrukturen wären ein stärkerer Robustheitscheck. Auch wiederholte synthetische Erfolge bleiben keine externe Validierung. Ein spätes Strukturbruch-/Drift-Szenario ist besonders hilfreich, weil es die Grenzen eines auf vergangenen Mustern trainierten Modells zeigt.

Die zusätzliche Bäckerei-Replikation ist eine eigenständige Folgeuntersuchung mit einem vor ihrem neuen Test festgelegten Startwert und einer zensierungsbewussten Spezifikation. Sie verändert das ursprüngliche negative Testresultat nicht. Verglichen werden ihre Regeln auf demselben neuen Nachfragepfad; Ergebnisse verschiedener Startwerte werden nicht als gepaarter Modellvergleich vermischt. Das Verfahren wurde als Reaktion auf eine erkannte methodische Grenze ergänzt. Deshalb ist die Replikation weder eine vorab registrierte Gesamtstudie noch ein ursprünglicher, völlig unberührter Nachweis der ursprünglichen Modellauswahl.

### Was in den Bäckereiversuchen tatsächlich berechnet wird

| Teil | Ursprünglicher Versuch | Eigenständige Folgeuntersuchung |
|---|---|---|
| Startwert | 20261014 | 2026101402; ursprüngliche Ergebnisdatei bleibt unverändert |
| Modellziel | Tatsächlich verkaufte Menge; zensierte Zeilen mit Gewicht 0,20 | Bedingte Nachfrageverteilung aus Dichte-/Survival-Likelihood beobachteter, gegebenenfalls zensierter Verkäufe |
| Modellwahl | Nicht ausverkaufter beobachteter Validierungs-MAE; HGB nur bei mindestens 1 % Vorteil gegenüber einfacheren Kandidaten | Zensierte Validierungs-Negativloglikelihood für Regularisierung; separate zensierte Kalibrierung für globalen Streuungsfaktor |
| Unsicherheit | Produktweise empirische standardisierte Residuen; nominelles p10-p90-Band | Quantile der angenommenen, kalibrierten bedingten Lognormalverteilung; keine verteilungsfreie Aussage |
| Hauptinterpretation | Verkaufsfehler für selektive nicht ausverkaufte Teilmenge | Beobachtbarer Fehler und nachgelagerter vollständiger Simulator-Nachfrageaudit getrennt |
| Mengenregel | Kostenquantil aus empirischer Residuenverteilung; gemeinsame Los-/Kapazitätsanpassung | Kostenquantil direkt aus der geschätzten Nachfrageverteilung; gleiche Los-/Kapazitätsanpassung |
| Wirtschaftsauswertung | Fixer historischer Informationsstand; keine geschlossene Verkaufsrückkopplung | Ebenfalls fixer Informationsstand, aber alle Vergleichsregeln auf demselben neuen Pfad |

Beide simulierten Testfenster liegen vom 1. Juli 2025 bis 30. Juni 2026. Gleiche Kalenderdaten machen unterschiedliche Zufallspfade nicht zu identischen Beobachtungen. Die latente Nachfrage bleibt bei Modellwahl und Kalibrierung in beiden Fällen unzugänglich. Der gespeicherte Optimiererstatus, Gradient und Kandidatenlog ermöglichen eine Prüfung der numerischen Schätzung; ein erfolgreicher Stopp ist kein Beweis für ein globales Optimum oder ein sachlich richtiges Verteilungsmodell.

## 12. Übertragung in einen realen Pilot

Vor einem Kundenprojekt werden Entscheidung, Horizont, Verantwortlicher, Datenzugriff und Handlungsraum festgelegt. Danach folgt ein enger Datencheck: stabile Kennungen, verfügbare Historie, tatsächlich beobachteter Verkauf versus Nachfrage, Bestände/Ausverkäufe, damalige Bestell-/Lieferzusagen und verlässliche Kosten.

Die erste reale Referenz ist die vorhandene Planung: Mensch, Excel, Warenwirtschaft oder bereits eingesetztes Modell. Ein Modellpilot lohnt sich erst, wenn zeitlich geprüfte Prognosen eine konkrete Entscheidung verbessern können. Zunächst läuft er parallel. Ein späterer, freigegebener Praxisversuch misst umgesetzte Mengen, tatsächliche Kosten und Lieferfähigkeit gegen einen geeigneten Vergleich; mögliche Interferenzen zwischen Artikeln und Standorten sind zu beachten.

Es gibt klare Abbruchgründe: kein rekonstruierbarer Stichtag, nur wenige heterogene Fälle, dominierende feste Lieferantenvorgaben, fehlende Einflussmöglichkeit, schwere Bestands-/Zeitfehler oder kein Zusatznutzen gegenüber einer guten einfachen Regel. Dann bleibt es bei der einfachen Regel oder einer Datenerfassungsverbesserung, statt ein ML-Projekt zu erzwingen.

## 13. Bericht und Reproduzierbarkeit

Der technische Bericht eröffnet mit dem tatsächlich berechneten Ergebnis und nennt dessen synthetischen Charakter. Danach zeigt er je Fall Modellvergleich, unberührte Testprognosen und Entscheidungsökonomie. Daten-/Metrikdefinitionen, Modell-/Splitdetails, Unsicherheit, Kostenannahmen und Grenzen stehen unmittelbar bei den zugehörigen Aussagen. Ein negativer oder gemischter Befund wird genauso gezeigt wie ein positiver.

Die Leserdatei ist offline portables HTML aus einem kanonischen Artefakt. Rohdateien, Auswahllogs, Quellen und Skripte bleiben im Projektordner. Der Bericht wird nicht aus ausgedachten Beispielkennzahlen aufgebaut. Fehlen Ergebnisse, endet die Erzeugung mit einer klaren Fehlermeldung; die fehlenden Werte werden nicht als Null dargestellt.

## 14. Was der Getränkefall konkret prüft

Der fiktive Betrieb hat zwei Filialen und acht Artikel je Filiale. Eine Verkaufseinheit ist eine Kiste oder ein Pack, keine einzelne Flasche. Das Modell schätzt beobachtete Verkäufe des am jeweiligen Stichtag noch nicht gebuchten Anteils. Bekannte Festaufträge gehen direkt in die Bestandsplanung; erst später registrierte Aufträge bleiben überraschender Bedarf. Das öffentlich beobachtbare Ausverkaufskennzeichen bedeutet **offener Verkaufstag und Schlussbestand null**. Es setzt nicht die geheime Kenntnis tatsächlich verlorener Nachfrage voraus; Bestandserschöpfung kann auch ohne entgangenen Verkauf auftreten. Verlorene Einheiten bleiben ausschließlich Simulator-/Wirtschaftsaudit. Im Fit und im Hauptfehlervergleich werden geschlossene und so gekennzeichnete Tage ausgeschlossen. Das ist eine selektive Teilmenge und keine vollständige Nachfragebeobachtung.

Die Zielhorizonte wechseln zwischen eins und acht Tagen. Deshalb genügt eine Trennung nach Zieltagen nicht: Der jeweilige Prognoseursprung darf nicht vor dem eingefrorenen Fit-/Kalibrierungsende liegen. Grenzzeilen werden entfernt. Parameterauswahl erfolgt im zweiten Halbjahr 2024 nach beobachtetem Validierungs-MAE. Die besten Ridge- und HGB-Familien werden anschließend bis 31.12.2024 neu geschätzt. Die getrennte Kalibrierung liegt im ersten Halbjahr 2025, die spätere Prognoseprüfung im folgenden Testjahr. HGB verwendet Poisson-Verlust und deaktiviertes internes Early Stopping. Es werden keine direkten Quantilbäume geschätzt.

Tagesintervalle stammen aus artikelspezifischen signierten Kalibrierungsfehlern. Für die Bestellung werden gemeinsame zusammenhängende Tages-/Artikel-Fehlerblöcke auf Prognosepfade gelegt; danach wird die Summe der Nachfragepfade gebildet und ihr Quantil genommen. Bekannte geschlossene Sonntage werden vor der Summenbildung auch in den Pfaden auf null gesetzt; Werktagsfehler dürfen dort keine künstliche Nachfrage erzeugen. Das erhält einen Teil der zeitlichen und gemeinsamen Artikelabhängigkeit. Die historische Matrix mischt allerdings Horizonte von eins bis acht. Fehlende und zensierte Residuen werden in diesen Pfadblöcken mit null ersetzt. **Diese Konstruktion ist eine Heuristik; sie kann Spitzen und Streuung unterschätzen und garantiert keinen 90%-Servicegrad.** Das nominelle 90%-Tagesintervall wird getrennt auf späteren beobachteten Verkäufen geprüft.

Bestellprüfung erfolgt mittwochs und samstags nach Ladenschluss. Lieferungen treffen nominell nach zwei Montag-bis-Freitag-Werktagen vor Öffnung ein; zusätzliche zufällige Verzögerungen von null bis zwei Werktagen sind am Bestellstichtag unbekannt. Die Schutzperiode umfasst Verkaufstage **vor** der nominalen nächsten Lieferung aus dem folgenden Prüfzeitpunkt; deren Ankunftstag ist eine exklusive Grenze, weil vor Öffnung angeliefert wird. Beispielsweise folgt auf Mittwoch, 14.10., die nächste Prüfung am Samstag, 17.10., und nominale Folgeankunft am Dienstag, 20.10. Der aktuelle nominale Schutzbedarf reicht deshalb vom 15. bis einschließlich 19.10.; Sonntag ist geschlossen und hat deterministisch null freien Bedarf. Die gemeinsame Kapazität von 1.650 Verkaufseinheiten umfasst vorhandene Ware und reservierte Pipeline. Bestellung wird auf Liefergebinde gerundet. Im Replay erfüllen Lieferungen zuerst gebuchten und anschließend freien Bedarf. Jede Regel aktualisiert ihre Vergangenheitsmerkmale aus eigenen erfüllten Verkäufen und Ausverkaufsflags; Kundenreaktionen, Preisänderungen und Lieferantenreaktionen auf die neue Regel werden nicht simuliert.

Die operative Rechnung umfasst realisierten Stückdeckungsbeitrag, Halte- und Bestellkosten. Haltekosten sind ausdrücklich 8 % jährliche Kapitalbelastung auf vorhandenen Warenwert plus 0,0035 EUR je vorhandener Einheit/Tag. Bestellungen kosten 10 EUR plus 0,40 EUR pro bestelltem Artikel. Fehlmargen werden gesondert gezeigt und nicht nochmals vom realisierten Deckungsbeitrag abgezogen. Endbestand bleibt Vermögen; offene Pipeline bleibt eine Verpflichtung. Es gibt eine zusätzliche Cashflow-Näherung bei sofortiger Kunden-/Lieferantenzahlung sowie Endbestands-/Pipelinewerte, aber keinen vollständigen gemessenen Erstjahres-Cashflow. Pfand und Umsatzsteuer sind ausgeschlossen.

Die Wirtschaftsreferenz ist die verfügbarkeitsbewusste saisonale Minmax-Regel mit denselben Buchungen, Gebinden, Kapazitäten und eigenem kalibriertem Sicherheitsverfahren. Die vorab beste einfache Prognosefamilie erhält außerdem einen eigenen Wirtschaftsreplay. Diese beiden Vergleiche werden nicht verwechselt. Eine Auswahl von Ridge ist ein gültiges Resultat, auch wenn HGB im späteren Test zufällig einen kleineren MAE erreichen sollte. Der Test wird dadurch nicht nachträglich zur Auswahlmenge.

Robustheit wird durch neue Replays für systematische 30%-Unterprognose, einen späteren Nachfragesprung und häufigere Lieferverzögerungen geprüft. Die reine Absenkung der Stückmarge auf 35 % bewertet dagegen unveränderte Mengen/Verkäufe neu. Anbieter-, Einführungs- und interner Prüfaufwand sind weitere arithmetische Kostenszenarien. Das gepaarte kreisförmige Moving-Block-Bootstrap verwendet 14-Tage-Blöcke; es erfasst nicht automatisch jahreszeitliche oder strukturelle Unsicherheit.

Das separate Zukunftsmodell ist mit Daten bis 31.07.2026 trainiert und mit späteren beobachtbaren Fehlern bis zum 14.10.2026 kalibriert. Die Bestellvorschläge vom 14.10.2026 ersetzen nur den noch nicht versandten Entwurf dieses Tages; ältere offene Bestellungen bleiben erhalten. Das ist eine reale Entscheidungssituation innerhalb des synthetischen Kalenders, keine bereits bestätigte künftige Ersparnis.
